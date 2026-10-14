/** Soft light behind everything, so the glass has something to bend. */
export function Backdrop() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="orb -top-[22vh] left-[4vw] size-[62vw]" />
      <div className="orb top-[30vh] -right-[18vw] size-[56vw]" />
      <div className="orb -bottom-[30vh] left-[26vw] size-[50vw]" />
    </div>
  )
}
