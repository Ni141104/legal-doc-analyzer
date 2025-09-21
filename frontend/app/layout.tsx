import './globals.css'
import { Inter } from 'next/font/google'
import { Toaster } from 'react-hot-toast'
import { AppProvider } from '@/contexts/AppContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Legal Document Analyzer MVP',
  description: 'AI-powered legal document analysis for hackathon prototype',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[var(--gemini-deep-blue)] min-h-screen antialiased`}>        
        <AppProvider>
          <main className="min-h-screen">
            {children}
          </main>
        </AppProvider>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'rgba(10, 14, 40, 0.8)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(138,145,180,0.25)',
              color: 'var(--starlight-white)',
              borderRadius: '14px',
              fontSize: '14px',
            },
          }}
        />
      </body>
    </html>
  )
}