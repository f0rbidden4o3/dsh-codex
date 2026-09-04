/** Durable request event owned by the OpenAI Codex search provider. */

import { createRequire } from 'node:module'
import { join, resolve } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { KNOWN_SESSION_EVENT_TYPES } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-agent'
import { resolveDshHome } from '@deepseek-ai/dsh-home-paths'
import type { OpenAICodexSearchRequestRecord } from './search.ts'

/** Dedicated log event written before an OpenAI Codex search dispatch. */
export const OPENAI_CODEX_SEARCH_MODEL_REQUEST_EVENT = 'web/openai-codex-search-llm-request'

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /** Exact secret-free OpenAI Codex standalone-search request. */
    'web/openai-codex-search-llm-request': OpenAICodexSearchRequestRecord
  }
}

interface SessionVocabularyModule {
  readonly KNOWN_SESSION_EVENT_TYPES?: unknown
}

const DSH_SESSION_PACKAGE = '@deepseek-ai/dsh-session'
const HOST_RESOLVER_FILENAME = '.dsh-codex-host-session-resolver.cjs'

function isMissingModule(error: unknown): boolean {
  return error instanceof Error
    && 'code' in error
    && error.code === 'MODULE_NOT_FOUND'
}

/** Resolve one session vocabulary from the package graph visible at `anchor`. */
function sessionVocabularyFrom(anchor: string): Set<string> | undefined {
  const require = createRequire(anchor)
  let entry: string
  try {
    entry = require.resolve(DSH_SESSION_PACKAGE)
  } catch (error) {
    if (isMissingModule(error)) return undefined
    throw error
  }
  const namespace = require(entry) as SessionVocabularyModule
  const vocabulary = namespace.KNOWN_SESSION_EVENT_TYPES
  if (!(vocabulary instanceof Set)) {
    throw new Error(`dsh-openai-codex: ${DSH_SESSION_PACKAGE} resolved from ${anchor} does not expose an extensible session event vocabulary`)
  }
  return vocabulary as Set<string>
}

/**
 * Register the plugin-owned event in every session-package instance that can
 * own the running Harness persistence reader. A normal installed plugin sees
 * the host package through `$DSH_HOME/profiles/node_modules`; an absolute
 * `link:` development install instead sees its own devDependency graph. The
 * two explicit host anchors keep that linked graph from registering only its
 * private Set. Registrations are process-lifetime additions so HMR cannot make
 * a session written before reload unreadable.
 */
export function installOpenAICodexSearchEvent(): void {
  if (!(KNOWN_SESSION_EVENT_TYPES instanceof Set)) {
    throw new Error('dsh-openai-codex: this plugin package does not expose an extensible session event vocabulary')
  }

  const vocabularies = new Set<Set<string>>([
    KNOWN_SESSION_EVENT_TYPES as Set<string>,
  ])
  const anchors = [
    join(resolveDshHome(), 'profiles', HOST_RESOLVER_FILENAME),
    ...(process.argv[1] === undefined ? [] : [resolve(process.argv[1])]),
  ]
  for (const anchor of anchors) {
    const vocabulary = sessionVocabularyFrom(anchor)
    if (vocabulary !== undefined) vocabularies.add(vocabulary)
  }
  for (const vocabulary of vocabularies) {
    vocabulary.add(OPENAI_CODEX_SEARCH_MODEL_REQUEST_EVENT)
  }
}

/**
 * Append one resolved request to the initiating agent's session. Searches
 * outside an agent turn have no owning session and therefore produce no log.
 * @param ctx - plugin context carrying the optional active-agent service.
 * @param request - exact request after defaults, excluding credentials.
 */
export function recordOpenAICodexSearchRequest(
  ctx: Context,
  request: OpenAICodexSearchRequestRecord,
): void {
  ctx.get('agents')?.currentInitiator()?.session.append(
    OPENAI_CODEX_SEARCH_MODEL_REQUEST_EVENT,
    request,
  )
}
