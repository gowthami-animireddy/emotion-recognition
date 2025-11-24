const nextConfig = {
  // <CHANGE> Added React Compiler and cache components for better performance
  reactCompiler: true,
  cacheComponents: true,
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
