# Prediction API

`GET /api/predict` returns service liveness. `POST /api/predict` accepts a JSON
object with finite, nonnegative numeric `recency`, `frequency`, and `monetary`
values. Send `Content-Type: application/json`; a charset parameter is accepted.
The request body is limited to 8 KB. Strings and booleans are not numeric inputs.

Successful predictions return `cluster`, `label`, and `profile`. Set
`generate_strategy` to `true` to request a marketing strategy. Its
`strategy_status` is `generated`, `not_configured`, or `unavailable`; the last two
return `strategy: null` while preserving the customer prediction.

| Status | Meaning |
| --- | --- |
| 200 | Prediction succeeded, or liveness check succeeded |
| 400 | Invalid JSON or invalid customer metrics |
| 413 | Empty body or body exceeds 8 KB |
| 415 | Missing or unsupported Content-Type |
| 503 | Model artifacts are unavailable |
| 500 | Unexpected internal prediction error |

Responses use `Cache-Control: no-store` and never contain API credentials.
