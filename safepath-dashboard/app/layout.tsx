import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import "leaflet/dist/leaflet.css";

export const metadata: Metadata = {
  title: "SafePath Dashboard",
  description: "NYC crash risk dashboard",
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
              <Link href="/" className="text-xl font-bold tracking-tight">
                SafePath
              </Link>

              <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
                <Link href="/" className="hover:text-slate-900">
                  Home
                </Link>
                <Link href="/crash-map" className="hover:text-slate-900">
                  Crash Map
                </Link>
                <Link href="/insights" className="hover:text-slate-900">
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