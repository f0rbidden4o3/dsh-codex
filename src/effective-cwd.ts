/** Optional structural adapter for DSH's live effective-working-directory service. */

import type { Context } from '@deepseek-ai/cordis'
import type { ToolExecution } from '@deepseek-ai/dsh-tools'

type Agent = NonNullable<ToolExecution['agent']>

type EffectiveWorkingDirectoryValue =
  | { kind: 'available'; cwd: string }
  | { kind: 'unavailable'; reason: string }

interface EffectiveWorkingDirectoryService {
  prepare(agent: Agent, signal: AbortSignal): Promise<unknown>
  resolve(agent: Agent): EffectiveWorkingDirectoryValue | undefined
}

/**
 * Prepare and resolve the calling agent's optional live working directory.
 * @param ctx - Plugin context that may expose the service supplied by newer DSH builds.
 * @param exec - Exact tool execution whose agent and abort signal must be prepared.
 * @returns The prepared cwd, the immutable session fallback, or undefined outside an agent cwd.
 */
export async function prepareEffectiveCwd(ctx: Context, exec: ToolExecution): Promise<string | undefined> {
  const agent = exec.agent
  if (agent === undefined) return undefined
  const service = ctx.get('effectiveWorkingDirectory') as EffectiveWorkingDirectoryService | undefined
  if (service === undefined) return agent.session.header.cwd
  await service.prepare(agent, exec.signal)
  const value = service.resolve(agent)
  if (value?.kind === 'unavailable') {
    throw new Error(`effective working directory is unavailable: ${value.reason}`)
  }
  return value?.kind === 'available' ? value.cwd : agent.session.header.cwd
}
