const API_BASE_URL = "http://127.0.0.1:5000";
const TOKEN_KEY = "sahaayToken";

async function request(path, { method = "GET", body, requiresAuth = false } = {}) {
  const headers = {
    "Content-Type": "application/json",
  };

  if (requiresAuth) {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      throw new Error("Please log in to continue.");
    }
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : {};

  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed with status ${response.status}.`);
  }

  return data;
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export async function loginUser(email, password) {
  return request("/login", {
    method: "POST",
    body: { email, password },
  });
}

export async function registerUser(name, email, password) {
  const response = await fetch(`${API_BASE_URL}/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ name, email, password })
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : {};

  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed with status ${response.status}.`);
  }

  return data;
}

export async function processAudio(text) {
  return request("/process_audio", {
    method: "POST",
    body: { text },
    requiresAuth: true,
  });
}
