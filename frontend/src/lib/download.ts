const escape = (cell: string | number) => {
  const text = String(cell)
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

/** Save rows as a CSV file. The BOM makes Excel read it as UTF-8. */
export function downloadCsv(filename: string, rows: (string | number)[][]) {
  const text = rows.map((row) => row.map(escape).join(",")).join("\n")
  const url = URL.createObjectURL(
    new Blob(["﻿" + text], { type: "text/csv;charset=utf-8" })
  )
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
