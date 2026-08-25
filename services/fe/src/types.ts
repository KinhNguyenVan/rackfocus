import type { SearchHit } from "./api/types";

export type Task = "kis" | "qa" | "trake";
export type Result = SearchHit & {
  url: string;
  video: string;
  frame: number;
};
