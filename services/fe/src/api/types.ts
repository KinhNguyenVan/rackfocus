// Khớp services/be/src/app/api/search.py — SearchRequest/Hit/SearchResponse.
export type SearchRequest = {
	text: string;
	top_k?: number;
	use_llm?: boolean;
	tags?: number[];
	min_score?: number;
	// false (mặc định) = "rerank": core tự chọn (HNSW+SQ8 coarse rồi rerank exact trên
	// top rerank_candidates, hoặc EXACT_SUBSET nếu tag đủ hẹp). true = "exact": ép
	// brute-force toàn candidate/corpus, bỏ qua HNSW hoàn toàn.
	exact?: boolean;
};

export type SearchHit = {
	point_id: number;
	row: number;
	score: number;
	rank: number;
	video_name: string;
	frame: number;
	keyframe_time: number;
	start_sec: number;
	end_sec: number;
	keyframe_url: string;
	clip_url: string;
	scene_idx: number;
	has_speech: boolean;
};

export type SearchResponse = {
	hits: SearchHit[];
	tags_used: number[];
	candidate_count: number;
	corpus_count: number;
	strategy: string;
	warnings: string[];
	snapshot_ver: string;
	timings_ms: Record<string, number>;
	enrichment: {
		model: string;
		tags: number[];
		enriched_text: string;
		error: string;
		used_llm: boolean;
	};
};

// Khớp services/be/src/app/api/browse.py -- GET /api/neighbors.
export type NeighborFrame = {
	url: string;
	frame: number;
	keyframe_time: number;
	scene_idx: number;
	start_sec: number;
	end_sec: number;
	clip_url: string;
	is_current: boolean;
};

export type NeighborsResponse = {
	video_name: string;
	current_frame: number;
	frames: NeighborFrame[];
	playback_source: "scene_clip";
};

// Khớp services/be/src/app/api/search_temporal.py.
export type TemporalSearchRequest = {
	event1: string;
	event2: string;
	use_llm?: boolean;
	exact?: boolean;
	top_k?: number;
};

export type TemporalChain = {
	video_name: string;
	score: number;
	span_sec: number;
	hits: SearchHit[]; // luôn đúng 2 phần tử, theo thứ tự event1 -> event2
};

export type TemporalSearchResponse = {
	chains: TemporalChain[];
	warnings: string[];
	tags_used: number[];
	snapshot_ver: string;
	timings_ms: Record<string, number>;
};
