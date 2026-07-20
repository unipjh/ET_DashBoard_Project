export default function Logo({ className = '' }: { className?: string }) {
  return (
    <div className={`inline-flex items-center gap-2.5 md:gap-3 ${className}`}>
      <span className="text-3xl md:text-5xl leading-none shrink-0" role="img" aria-label="newspaper">📰</span>
      <span className="leading-[1.05] select-none">
        <span className="block font-extrabold uppercase tracking-tight text-red-500 text-lg md:text-2xl">
          Trusted
        </span>
        <span className="block font-extrabold uppercase tracking-tight text-navy-600 text-lg md:text-2xl -mt-1">
          News
        </span>
      </span>
    </div>
  )
}
