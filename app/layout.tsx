import type {Metadata} from "next";
import "./globals.css";
import RegisterSW from "./register-sw";
export const metadata:Metadata={title:"Gyaan Saathi",description:"Offline-first learning diagnostics for low-connectivity classrooms",manifest:"/manifest.webmanifest"};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body><RegisterSW/>{children}</body></html>;}