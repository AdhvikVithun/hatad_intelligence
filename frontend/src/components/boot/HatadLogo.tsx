/**
 * HATAD Logo — uses the official brand PNG asset.
 * Wrapped in a div so GSAP can target .hatad-leaf / .hatad-wordmark
 * classes for animation if needed.
 */
export function HatadLogo({ className = '' }: { className?: string }) {
  return (
    <div className={className}>
      <img
        src="/hatad-logo.png"
        alt="HATAD"
        className="hatad-logo-img"
        draggable={false}
      />
    </div>
  );
}
