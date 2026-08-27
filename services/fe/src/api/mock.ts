import type {
	NeighborFrame,
	NeighborsResponse,
	SearchHit,
	SearchResponse,
	SearchRequest,
	TemporalSearchRequest,
	TemporalSearchResponse,
} from "./types";

export const MOCK_KEYFRAME_URLS: string[] = [
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000000.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000001.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000002.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000004.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000005.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000006.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000008.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000014.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000019.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000025.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000031.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000036.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000042.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000048.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000088.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000124.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000161.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000198.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000234.webp",
	"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/Keyframes_L21_a/keyframes/L21_V001/000271.webp",
] as const;

const VIDEO_NAME = "L21_V001";

function frameFromUrl(url: string): number {
	return Number(url.match(/(\d+)\.webp(?:$|\?)/)?.[1] ?? 0);
}

function hit(url: string, rank: number): SearchHit {
	const frame = frameFromUrl(url);
	const time = frame / 30;
	return {
		point_id: rank + 1,
		row: rank,
		score: 1 - rank * 0.02,
		rank,
		video_name: VIDEO_NAME,
		frame,
		keyframe_time: time,
		start_sec: time,
		end_sec: time + 1,
		keyframe_url: url,
		clip_url: "",
		scene_idx: rank,
		has_speech: false,
	};
}

export function mockSearch(request: SearchRequest): SearchResponse {
	const hits = MOCK_KEYFRAME_URLS.slice(0, request.top_k ?? 20).map(hit);
	return {
		hits,
		tags_used: [],
		candidate_count: MOCK_KEYFRAME_URLS.length,
		corpus_count: MOCK_KEYFRAME_URLS.length,
		strategy: "mock",
		warnings: ["frontend_mock"],
		snapshot_ver: "mock",
		timings_ms: { total: 1 },
		enrichment: {
			model: "mock",
			tags: [],
			enriched_text: request.text,
			error: "",
			used_llm: false,
		},
	};
}

export function mockNeighbors(keyframeUrl: string, before: number, after: number): NeighborsResponse {
	const currentIndex = Math.max(0, MOCK_KEYFRAME_URLS.indexOf(keyframeUrl));
	const start = Math.max(0, currentIndex - before);
	const end = Math.min(MOCK_KEYFRAME_URLS.length, currentIndex + after + 1);
	return {
		video_name: VIDEO_NAME,
		current_frame: frameFromUrl(MOCK_KEYFRAME_URLS[currentIndex]),
		playback_source: "scene_clip",
		frames: MOCK_KEYFRAME_URLS.slice(start, end).map((url, index) => {
			const frame = frameFromUrl(url);
			return {
				url,
				frame,
				keyframe_time: frame / 30,
				scene_idx: start + index,
				start_sec: frame / 30,
				end_sec: frame / 30 + 1,
				clip_url: "",
				is_current: url === keyframeUrl,
			};
		}),
	};
}

export function mockTemporalSearch(request: TemporalSearchRequest): TemporalSearchResponse {
	const first = hit(MOCK_KEYFRAME_URLS[2], 0);
	const second = hit(MOCK_KEYFRAME_URLS[12], 1);
	return {
		chains: [{ video_name: VIDEO_NAME, score: 0.96, span_sec: second.keyframe_time - first.keyframe_time, hits: [first, second] }],
		warnings: ["frontend_mock", `${request.event1} -> ${request.event2}`],
		tags_used: [],
		snapshot_ver: "mock",
		timings_ms: { total: 1 },
	};
}