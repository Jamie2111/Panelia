/** @type {import('next').NextConfig} */
const backendTarget = process.env.PANELIA_BACKEND_PROXY_TARGET ?? "http://127.0.0.1:8000";

const nextConfig = {
  typedRoutes: true,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${backendTarget}/:path*`
      },
      {
        source: "/media/:path*",
        destination: `${backendTarget}/media/:path*`
      },
      {
        source: "/assets/:path*",
        destination: `${backendTarget}/assets/:path*`
      }
    ];
  }
};

export default nextConfig;
