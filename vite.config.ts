import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins:[react(),VitePWA({
    registerType:"autoUpdate",
    workbox:{runtimeCaching:[
      {urlPattern:/\/api\/v1\/diagnostic\/questions.*/,handler:"StaleWhileRevalidate"},
      {urlPattern:/\/api\/v1\/likert\/items.*/,handler:"StaleWhileRevalidate"}
    ]},
    manifest:{name:"Gyaan Saathi",short_name:"Gyaan Saathi",start_url:"/",display:"standalone",theme_color:"#0f766e",background_color:"#ffffff"}
  })]
});