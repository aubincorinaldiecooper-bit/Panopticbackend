import { env } from '../../../config/env.js';
import type { VideoModelAdapter } from '../model.js';
import { miniCpmWatcherAdapter } from './minicpm.js';
import { videoChat3Adapter } from './videochat3.js';

/**
 * The model that watches internet videos for the live search's scouts.
 *
 * MiniCPM-V 4.6 on the team's own GPU server by default (the owner's choice,
 * 2026-11-26); VideoChat3 on Modal while it stays selectable. Decided once
 * per search, so one search never mixes watchers.
 */
export function internetWatcher(): VideoModelAdapter {
  return env.INTERNET_WATCHER === 'videochat3' ? videoChat3Adapter : miniCpmWatcherAdapter;
}
