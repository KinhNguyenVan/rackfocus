import type { NeighborsResponse, SearchRequest, SearchResponse, TemporalSearchRequest, TemporalSearchResponse } from "./types";

export async function search(request: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
	const response = await fetch("/api/search", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(request),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Search failed (${response.status})`);
	}

	return response.json() as Promise<SearchResponse>;
}

export async function neighbors(
	keyframeUrl: string,
	before = 25,
	after = 25,
	toKey?: string,
	signal?: AbortSignal,
): Promise<NeighborsResponse> {
	const params = new URLSearchParams({
		key: keyframeUrl,
		before: String(before),
		after: String(after),
	});
	if (toKey) params.set("to_key", toKey);
	const response = await fetch(`/api/neighbors?${params}`, { signal });

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Neighbors failed (${response.status})`);
	}

	return response.json() as Promise<NeighborsResponse>;
}

export async function searchTemporal(
	request: TemporalSearchRequest,
	signal?: AbortSignal,
): Promise<TemporalSearchResponse> {
	const response = await fetch("/api/search/temporal", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(request),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Temporal search failed (${response.status})`);
	}

	return response.json() as Promise<TemporalSearchResponse>;
}
