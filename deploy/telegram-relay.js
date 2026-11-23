// Cloudflare Worker that forwards Bot API calls to Telegram.
// Use it only if Telegram blocks or rate-limits your server's IP (see README, "Telegram relay").
// It forwards bot API and file paths only, and stores nothing.
export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/bot") && !url.pathname.startsWith("/file/bot")) {
      return new Response("Not found", { status: 404 });
    }
    const target = "https://api.telegram.org" + url.pathname + url.search;
    return fetch(new Request(target, request));
  },
};
