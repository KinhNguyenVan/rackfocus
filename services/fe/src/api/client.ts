import type { NeighborsResponse, SearchRequest, SearchResponse } from "./types";

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
	signal?: AbortSignal,
): Promise<NeighborsResponse> {
	const params = new URLSearchParams({
		key: keyframeUrl,
		before: String(before),
		after: String(after),
	});
	const response = await fetch(`/api/neighbors?${params}`, { signal });

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Neighbors failed (${response.status})`);
	}

	return response.json() as Promise<NeighborsResponse>;
}
