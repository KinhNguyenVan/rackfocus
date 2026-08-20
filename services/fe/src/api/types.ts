export type SearchRequest = {
	text: string;
	top_k?: number;
};

export type SearchHit = {
	scene_id: number;
	score: number;
	rank: number;
	url?: string;
};

export type SearchResponse = {
	hits: SearchHit[];
	timings?: { total_ms?: number };
	snapshot?: string;
};
