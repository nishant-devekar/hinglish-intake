// On-screen text from frames, using macOS's built-in Vision OCR (no model
// download, runs on the Neural Engine).
//
//   swift bin/ocr.swift out/<name>/frames > out/<name>/frames/ocr.txt
//
// One block per frame: "== frame-0001.jpg" then the recognised lines, top to
// bottom. Screen recordings carry most of their meaning as on-screen text, and
// text is far cheaper to search and quote than images.
import Foundation
import Vision
import AppKit

let dir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ".")
let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
var previous = ""

for url in files.filter({ $0.pathExtension.lowercased() == "jpg" }).sorted(by: { $0.path < $1.path }) {
    guard let img = NSImage(contentsOf: url),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = ["en-US", "hi-IN"]
    try? VNImageRequestHandler(cgImage: cg).perform([req])
    let lines = (req.results ?? [])
        .sorted { $0.boundingBox.minY > $1.boundingBox.minY }
        .compactMap { $0.topCandidates(1).first }
        .filter { $0.confidence > 0.4 }
        .map { $0.string }
    let text = lines.joined(separator: "\n")
    // Consecutive frames of a static screen repeat the same text; say so
    // instead of printing it again.
    print("== \(url.lastPathComponent)")
    print(text.isEmpty ? "(no text)" : text == previous ? "(same as previous)" : text)
    previous = text
}
