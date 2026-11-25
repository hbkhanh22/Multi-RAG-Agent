/** Product mark: three retrieval branches converging into one answer node. */
export default function LogoMark({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="5" cy="5" r="2.25" fill="currentColor" />
      <circle cx="5" cy="19" r="2.25" fill="currentColor" />
      <circle cx="5" cy="12" r="2.25" fill="currentColor" />
      <path d="M7 5.5 L14.5 11 M7 12 H14.5 M7 18.5 L14.5 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="18" cy="12" r="3.5" fill="currentColor" />
    </svg>
  );
}
