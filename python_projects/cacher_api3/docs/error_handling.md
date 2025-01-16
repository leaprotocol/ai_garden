# Error Handling

The Cacher API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (e.g., invalid input)
- `404`: Page Not Found (e.g., page not found). Important: When a state is not found, the API will return a 400 error.
- `500`: Internal Server Error

Errors include detailed messages and are logged for debugging.
