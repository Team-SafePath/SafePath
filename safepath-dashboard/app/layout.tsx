import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import "./globals.css";
import "leaflet/dist/leaflet.css";

export const metadata: Metadata = {
  title: "SafePath Dashboard",
  description: "NYC crash risk dashboard",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-white text-slate-900">
        <div className="min-h-screen">
          <header className="border-b border-slate-200 bg-white">
            <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
              
              {/* Logo + Title */}
              <Link href="/" className="flex items-center gap-2">
                <Image
                  src="/icon.svg"
                  alt="SafePath logo"
                  width={32}
                  height={32}
                />

                <span className="text-slate-900 font-semibold tracking-tight">
                  Safe<span className="text-indigo-600">Path</span>
                </span>
              </Link>

              {/* Nav */}
              <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
                <Link href="/" className="hover:text-slate-900 transition">
                  Home
                </Link>
                <Link href="/crash-map" className="hover:text-slate-900 transition">
                  Crash Map
                </Link>
                <Link href="/insights" className="hover:text-slate-900 transition">
                  Insights
                </Link>
              </nav>
            </div>
          </header>

          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}