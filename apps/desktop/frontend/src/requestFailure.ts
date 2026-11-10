import { ApiError } from "@wenyi/ui/http";
import routeWords from "../../diagnostic-routes.json";

/** How a request failed, in terms the native validator accepts: classes, never text. */
export type RequestFailure = {
  method: string;
  route: string;
  /** Absent when the engine answered: `status_code` then says what happened. */
  category?: "timeout" | "interrupted" | "io" | "invalid" | "unknown";
  status_code?: number;
  request_id?: string;
  error_id?: string;
};

const WORDS = new Set<string>(routeWords);
/** Words followed by user text: a glossary source or a connection name. */
const NAMES_FOLLOW = new Set(["terms", "credentials"]);
const IDENTITY = /^[0-9a-f]{32}$/;

/** Reduce a request path to its route shape.
 *
 * A path carries project identities and can carry a glossary term, so every
 * segment that is not a registered API word becomes `:id`, and the query is
 * dropped. A segment where user text can stand is always `:id`, even when it
 * spells an API word: a term named "page" must not be recorded as `page`.
 * The native side checks the result against the same rules.
 */
export function routeShape(path: string): string {
  const segments = path.split(/[?#]/, 1)[0].split("/").filter(Boolean);
  const shape = segments.map((segment, index) =>
    WORDS.has(segment) && !NAMES_FOLLOW.has(segments[index - 1]) ? segment : ":id");
  return `/${shape.join("/")}`;
}

/** A deadline or a cancellation, whichever phase of the request it stopped. */
function interruption(error: unknown): RequestFailure["category"] {
  // The shared client has already told an expired deadline apart from a cancellation.
  if (error instanceof ApiError && error.code === "request_timeout") return "timeout";
  // The caller cancelled, usually because the view that asked went away.
  return error instanceof DOMException && error.name === "AbortError" ? "interrupted" : undefined;
}

/** Describe a failed request without its path, body, message or response text. */
export function requestFailure(
  path: string,
  method: string | undefined,
  error: unknown,
  response: Response | undefined,
): RequestFailure {
  const failure: RequestFailure = {
    method: (method ?? "GET").toUpperCase(),
    route: routeShape(path),
  };
  const stopped = interruption(error);
  if (stopped) failure.category = stopped;
  // `fetch` rejects with a TypeError when no response arrived at all.
  else if (!response) failure.category = error instanceof TypeError ? "io" : "unknown";
  // An answer that could not be read is a different failure from a refusal.
  else if (response.ok) failure.category = "invalid";
  if (!response) return failure;
  if (!response.ok) failure.status_code = response.status;
  for (const [field, header] of [["request_id", "X-Request-ID"], ["error_id", "X-Error-ID"]] as const) {
    const value = response.headers.get(header);
    if (value && IDENTITY.test(value)) failure[field] = value;
  }
  return failure;
}
