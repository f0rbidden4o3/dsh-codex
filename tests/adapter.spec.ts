import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AttachmentStore,
  ImageAttachmentRef,
} from "@deepseek-ai/dsh-attachment";
import { createAssistantMessage } from "@deepseek-ai/dsh-llm";
import type { GenerateOptions, StreamChunk } from "@deepseek-ai/dsh-llm";
import { PiAiAdapter } from "@deepseek-ai/dsh-llm-pi-ai";
import type { OpenAICodexCredentialStore } from "../src/store.ts";
import { OPENAI_CODEX_PROVIDER } from "../src/store.ts";
import {
  createOpenAICodexAdapter,
  openAICodexModelCatalog,
  OPENAI_CODEX_HIGH_DETAIL_MAX_PATCHES,
  OPENAI_CODEX_HIGH_DETAIL_MAX_DIMENSION,
  OPENAI_CODEX_IMAGE_PATCH_SIZE,
  OPENAI_CODEX_PROMPT_IMAGE_INPUT_GUARD_BYTES,
  OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
  OPENAI_CODEX_RETRY_POLICY,
  openAICodexRequestImagePixelBudget,
} from "../src/adapter.ts";
import { Config } from "../src/index.ts";
import { OpenAICodexResponseRuntime } from "../src/responses.ts";

type PreparedCallLike = {
  readonly model: Awaited<ReturnType<PiAiAdapter["resolveModel"]>>;
  readonly stream: (options: GenerateOptions) => AsyncIterable<StreamChunk>;
};

type PrepareCallLike = (
  provider: string,
  model: string,
  signal?: AbortSignal
) => Promise<PreparedCallLike>;

const piAiAdapterPrototype = PiAiAdapter.prototype as unknown as {
  prepareCall?: PrepareCallLike;
};
const originalPrepareCall = Object.getOwnPropertyDescriptor(
  PiAiAdapter.prototype,
  "prepareCall"
);

afterEach(() => {
  vi.restoreAllMocks();
  if (originalPrepareCall === undefined) delete piAiAdapterPrototype.prepareCall;
  else
    Object.defineProperty(
      PiAiAdapter.prototype,
      "prepareCall",
      originalPrepareCall
    );
});

describe("OpenAI Codex adapter policy", () => {
  it("validates optional catalog and context-window configuration", () => {
    expect(Config({}).models).toBeUndefined();
    expect(Config({}).contextWindow).toBeUndefined();
    expect(Config({}).overrideSparkContextWindow).toBe(false);
    expect(
      Config({
        models: [],
        contextWindow: 512_000,
        overrideSparkContextWindow: true,
      })
    ).toMatchObject({
      models: [],
      contextWindow: 512_000,
      overrideSparkContextWindow: true,
    });
    expect(() => Config({ contextWindow: 0 })).toThrow();
  });

  it("supplies the complete request-image policy required by current DSH runtimes", () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false })
    );
    const profile = (
      adapter as unknown as {
        config: { profiles: () => Map<string, Record<string, unknown>> };
      }
    ).config
      .profiles()
      .get(OPENAI_CODEX_PROVIDER);

    expect(profile).toMatchObject({
      maxRequestImageBytes: OPENAI_CODEX_PROMPT_IMAGE_INPUT_GUARD_BYTES,
      requestImagePixelBudget: OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
      requestImageMaxBytes: OPENAI_CODEX_PROMPT_IMAGE_INPUT_GUARD_BYTES,
    });
    expect(OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET).toBe(
      OPENAI_CODEX_HIGH_DETAIL_MAX_PATCHES * OPENAI_CODEX_IMAGE_PATCH_SIZE ** 2
    );
  });

  it("combines the Codex longest-edge and rounded patch-grid image limits", () => {
    const cases: readonly {
      source: readonly [number, number];
      expected?: readonly [number, number];
    }[] = [
      { source: [2048, 2048], expected: [1600, 1600] },
      { source: [8192, 512], expected: [2048, 128] },
      { source: [1599, 1601] },
      { source: [1200, 800], expected: [1200, 800] },
    ];

    for (const { source, expected } of cases) {
      const [width, height] = source;
      const budget = openAICodexRequestImagePixelBudget(
        width,
        height,
        OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET
      );
      const scale = Math.min(1, Math.sqrt(budget / (width * height)));
      const projectedWidth =
        width >= height
          ? Math.max(1, Math.floor(width * scale))
          : Math.max(
              1,
              Math.round(
                (Math.max(1, Math.floor(height * scale)) * width) / height
              )
            );
      const projectedHeight =
        width >= height
          ? Math.max(1, Math.round((projectedWidth * height) / width))
          : Math.max(1, Math.floor(height * scale));
      expect(projectedWidth).toBeLessThanOrEqual(
        OPENAI_CODEX_HIGH_DETAIL_MAX_DIMENSION
      );
      expect(projectedHeight).toBeLessThanOrEqual(
        OPENAI_CODEX_HIGH_DETAIL_MAX_DIMENSION
      );
      expect(
        Math.ceil(projectedWidth / OPENAI_CODEX_IMAGE_PATCH_SIZE) *
          Math.ceil(projectedHeight / OPENAI_CODEX_IMAGE_PATCH_SIZE)
      ).toBeLessThanOrEqual(OPENAI_CODEX_HIGH_DETAIL_MAX_PATCHES);
      if (expected !== undefined)
        expect([projectedWidth, projectedHeight]).toEqual(expected);
    }
  });

  it("applies the Codex dimension projection at the attachment-store boundary", async () => {
    const readImageRequest = vi.fn().mockResolvedValue({});
    const store = { readImageRequest } as unknown as AttachmentStore;
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => store,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false })
    );
    const resolved = (
      adapter as unknown as {
        config: { resolveAttachments: () => AttachmentStore | undefined };
      }
    ).config.resolveAttachments();
    const ref = { width: 8192, height: 512 } as ImageAttachmentRef;

    await resolved?.readImageRequest(ref, {
      maxPixels: OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
      maxBytes: OPENAI_CODEX_PROMPT_IMAGE_INPUT_GUARD_BYTES,
    });

    expect(readImageRequest).toHaveBeenCalledWith(
      ref,
      {
        maxPixels: 2048 * 128,
        maxBytes: OPENAI_CODEX_PROMPT_IMAGE_INPUT_GUARD_BYTES,
      },
      undefined
    );
  });

  it("registers the extended bounded retry policy on the provider route", () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false })
    );

    expect(adapter.providerRetryPolicy(OPENAI_CODEX_PROVIDER)).toBe(
      OPENAI_CODEX_RETRY_POLICY
    );
    expect(OPENAI_CODEX_RETRY_POLICY).toMatchObject({
      mode: "normal",
      maxRetries: 5,
      retryableCodes: expect.arrayContaining([
        "RATE_LIMIT",
        "SERVER",
        "TIMEOUT",
        "TRANSPORT",
      ]),
      initialDelayMs: 1_000,
      maxDelayMs: 30_000,
      jitterRatio: 0.2,
    });
  });

  it("preserves compaction purpose around snapshot-bound prepared streams", async () => {
    const release = vi.fn();
    const enterCompaction = vi
      .spyOn(OpenAICodexResponseRuntime.prototype, "enterCompaction")
      .mockReturnValue(release);
    let dispatched: GenerateOptions | undefined;
    const downstream: PreparedCallLike["stream"] = (options) =>
      (async function* () {
        dispatched = options;
      })();
    const snapshotModel: PreparedCallLike["model"] = {
      provider: OPENAI_CODEX_PROVIDER,
      id: "gpt-5.6-sol",
      name: "GPT-5.6 Sol",
    };
    const prepareCall = vi.fn(async () => ({
      model: snapshotModel,
      stream: downstream,
    }));
    piAiAdapterPrototype.prepareCall = prepareCall;
    const directStream = vi.spyOn(PiAiAdapter.prototype, "stream");
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: true })
    );
    const options: GenerateOptions = {
      provider: OPENAI_CODEX_PROVIDER,
      model: "gpt-5.6-sol",
      purpose: "compaction",
      sessionId: "session-prepared-compaction" as never,
      messages: [
        createAssistantMessage({
          content: [{ type: "text", text: "legacy response" }],
          source: {
            provider: OPENAI_CODEX_PROVIDER,
            model: "gpt-5.6-sol",
            replayState: {
              kind: "pi-ai",
              version: 1,
              api: "openai-codex-responses",
              provider: OPENAI_CODEX_PROVIDER,
              model: "gpt-5.6-sol",
              stopReason: "stop",
              blocks: [],
            },
          },
        }),
      ],
    };

    const prepared = await (
      adapter as PiAiAdapter & { prepareCall: PrepareCallLike }
    ).prepareCall(OPENAI_CODEX_PROVIDER, "gpt-5.6-sol");
    expect(prepareCall).toHaveBeenCalledOnce();
    expect(prepareCall).toHaveBeenCalledWith(
      OPENAI_CODEX_PROVIDER,
      "gpt-5.6-sol",
      undefined
    );
    expect(prepared.model).toBe(snapshotModel);
    for await (const _chunk of prepared.stream(options)) {
      // The fake snapshot-bound stream intentionally emits no chunks.
    }

    expect(enterCompaction).toHaveBeenCalledOnce();
    expect(enterCompaction).toHaveBeenCalledWith(
      "session-prepared-compaction"
    );
    expect(release).toHaveBeenCalledOnce();
    expect(directStream).not.toHaveBeenCalled();
    const source = dispatched?.messages[0]?.source;
    expect(source?.kind).toBe("model");
    if (source?.kind !== "model")
      throw new Error("expected a migrated model source");
    expect(source.replayState).toEqual({
      response: {
        kind: "pi-ai",
        version: 2,
        api: "openai-codex-responses",
        provider: OPENAI_CODEX_PROVIDER,
        model: "gpt-5.6-sol",
        stopReason: "stop",
      },
      blocks: [],
    });
  });

  it("advertises only configured models while keeping hidden models resolvable", async () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
      undefined,
      () => ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-terra"]
    );

    const models = await adapter.listModels(OPENAI_CODEX_PROVIDER);
    expect(models.map((model) => model.id)).toEqual([
      "gpt-5.6-luna",
      "gpt-5.6-terra",
    ]);

    await expect(
      adapter.resolveModel(OPENAI_CODEX_PROVIDER, "gpt-5.4")
    ).resolves.toMatchObject({
      provider: OPENAI_CODEX_PROVIDER,
      id: "gpt-5.4",
    });
  });

  it("rotates snapshot-consistent profiles when the client-side capacity changes", async () => {
    let contextWindow: number | null = null;
    let overrideSparkContextWindow = false;
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
      undefined,
      undefined,
      () => contextWindow,
      () => overrideSparkContextWindow
    );

    const profileLoader = (
      adapter as unknown as {
        config: {
          profiles(): Map<
            string,
            {
              piProvider: {
                getModels(): readonly { id: string; contextWindow: number }[];
              };
            }
          >;
        };
      }
    ).config.profiles;
    const firstProfiles = profileLoader();
    await expect(
      adapter.resolveModel(OPENAI_CODEX_PROVIDER, "gpt-5.6-sol")
    ).resolves.toMatchObject({
      context: { contextWindow: 272_000 },
    });

    contextWindow = 512_000;
    const secondProfiles = profileLoader();
    expect(secondProfiles).not.toBe(firstProfiles);
    expect(profileLoader()).toBe(secondProfiles);
    expect(
      firstProfiles
        .get(OPENAI_CODEX_PROVIDER)
        ?.piProvider.getModels()
        .find((model) => model.id === "gpt-5.6-sol")
    ).toMatchObject({
      contextWindow: 272_000,
    });
    expect(
      secondProfiles
        .get(OPENAI_CODEX_PROVIDER)
        ?.piProvider.getModels()
        .find((model) => model.id === "gpt-5.6-sol")
    ).toMatchObject({
      contextWindow: 512_000,
    });
    await expect(
      adapter.resolveModel(OPENAI_CODEX_PROVIDER, "gpt-5.6-sol")
    ).resolves.toMatchObject({
      context: { contextWindow: 512_000 },
    });
    await expect(
      adapter.resolveModel(OPENAI_CODEX_PROVIDER, "gpt-5.3-codex-spark")
    ).resolves.toMatchObject({
      context: { contextWindow: 128_000 },
    });

    overrideSparkContextWindow = true;
    const thirdProfiles = profileLoader();
    expect(thirdProfiles).not.toBe(secondProfiles);
    await expect(
      adapter.resolveModel(OPENAI_CODEX_PROVIDER, "gpt-5.3-codex-spark")
    ).resolves.toMatchObject({
      context: { contextWindow: 512_000 },
    });
  });

  it("projects provider context capacities into the settings catalog", () => {
    expect(
      openAICodexModelCatalog().find((model) => model.id === "gpt-5.6-sol")
    ).toMatchObject({
      contextWindow: 272_000,
    });
  });

  it("advertises the full provider catalog when no model list is configured", async () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false })
    );

    const models = await adapter.listModels(OPENAI_CODEX_PROVIDER);
    expect(models.map((model) => model.id)).toEqual(
      expect.arrayContaining([
        "gpt-5.4",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
      ])
    );
  });
});
