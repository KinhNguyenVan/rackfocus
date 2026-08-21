import type { SearchRequest, SearchResponse } from "./types";

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
