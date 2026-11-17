# Prompt for Google AI Studio Build Mode

Create a responsive Spanish-language web app named **BasinCast** to forecast a user-uploaded monthly water time series. The visual interface should be clean and scientific, not a chat application.

## Architecture and mandatory rules

- React/TypeScript frontend with a Node.js server proxy. No direct calls from the browser to the Python forecasting service.
- The forecasting engine already exists as an external FastAPI service. Do NOT reimplement, emulate, approximate, or invent any forecast, KGE, RMSE, selected model, or reliability horizon in React/Node. Display only values returned by the external API.
- Access external service URL from `process.env.BASINCAST_API_URL` and secret from `process.env.BASINCAST_API_KEY` **server-side only**. Never expose the secret in browser bundles, HTML, Vite public environment variables or browser network requests.
- Do NOT use the Gemini API, a paid LLM, a database, or any paid service for these calculations. This app needs no AI inference at runtime.

## Form

- A file input accepting `.csv` and `.xlsx` (max 5 MiB), with a CSV sample format `date,value` (e.g., `2020-01-01,12.3`).
- `series_id`: safe identifier, default `Site_001`.
- `asset_type`: river, aquifer, reservoir, agricultural_demand, urban_demand, custom.
- `unit`: required free-text physical unit, never silently convert units.
- `max_horizon_months`: 1-48, default 12.
- Optional `date_column`, `value_column`, `aggregation` (mean/sum/last), `decimal_comma`, `capacity` for reservoirs only.
- Indicate clearly: the user-provided target is the forecast target. Historical river level does not determine agricultural/urban demand by itself.

## Backend proxy

- Implement `/api/forecast` (POST multipart) in the Node backend: take the uploaded file and form fields, forward as `multipart/form-data` to `${process.env.BASINCAST_API_URL}/v1/forecast`, and set HTTP header `X-API-Key` to the server-side secret. Return the FastAPI JSON status code and body without changing metric values.
- Implement `/api/forecast/download` (POST multipart) with the same inputs, forwarding to `/v1/forecast.zip`; stream the returned ZIP to the browser with `Content-Type: application/zip`.
- Set explicit timeouts compatible with cold starts of a free host, show a friendly retry message if the hosting service is waking up. Do not retry non-idempotent requests without user confirmation.
- Sanitize and cap uploads; forward helpful validation errors from the API. Do not log uploaded observations or secrets.
- Add a small health indicator backed by the API `/health` endpoint, not a fabricated status.

## Results

- Show issued date, the variable and units, n_calendar_months and the continuous supported horizon from `response.summary`.
- Plot `response.forecast` as candidate horizon forecasts, and visually distinguish months with `is_recommended === true` from exploratory candidates.
- Never label all candidate months as reliable. `response.recommended_forecast` is the only supported continuous trajectory.
- Provide a horizon table from `response.metrics` showing selected_model, `kge_validation`, `kge`, `n_test_pairs`, `n_disjoint_origin_windows_proxy`, `quality_supported`, `rmse`, `rmse_skill_vs_validation_selected_baseline` and `reliability_label`.
- Explain the difference between absolute KGE quality and incremental RMSE skill of Ridge. Avoid statements about calibrated uncertainty intervals; the engine returns point forecasts only.
- Include a button to download the original scientific ZIP via the backend proxy.
- Provide a simple example CSV snippet and precise error handling.

Do not add stochastic climate generators, IPCC-derived hydrological forecasts, extra machine-learning algorithms, invented demand observations or fictional confidence bands.
