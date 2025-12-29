import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Credit Intelligence Studio',
  description: 'AI-Powered Credit Risk Assessment',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-studio-bg text-studio-text">
        {children}
      </body>
    </html>
  )
}
