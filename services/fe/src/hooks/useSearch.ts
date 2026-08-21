import { useEffect, useRef, useState } from "react";
import { search } from "../api/client";
import type { SearchHit } from "../api/types";

export function useSearch(query: string, topK = 10) {
	const [hits, setHits] = useState<SearchHit[]>([]);
	const [totalMs, setTotalMs] = useState<number | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const requestId = useRef(0);

	useEffect(() => {
		const text = query.trim();
		if (!text) {
			setHits([]);
			setTotalMs(null);
			setError(null);
			setLoading(false);
			return;
		}

		const controller = new AbortController();
		const currentRequest = ++requestId.current;
		setLoading(true);
		setError(null);

		search({ text, top_k: topK }, controller.signal)
			.then((data) => {
				if (currentRequest !== requestId.current) return;
				setHits(data.hits ?? []);
				setTotalMs(data.timings_ms?.total ?? null);
			})
			.catch((cause: unknown) => {
				if (controller.signal.aborted || currentRequest !== requestId.current) return;
				setError(cause instanceof Error ? cause.message : "Search failed");
				setHits([]);
				setTotalMs(null);
			})
			.finally(() => {
				if (currentRequest === requestId.current) setLoading(false);
			});

		return () => controller.abort();
	}, [query, topK]);

	return { hits, totalMs, error, loading };
}
