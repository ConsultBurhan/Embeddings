File bytes arrive
│
├── PyMuPDF → extract text per page (DONE)
│    └── empty page? → OCR fallback
│
├── PyMuPDF → extract images (DONE)
│    └── GPT-4o → describe each image (DONE)
│
├── Chunk text pages individually
├── Keep image descriptions as individual chunks
│
└── Embed all chunks (text + image desc) separately
     └── Store in Qdrant with metadata
          ├── file_url
          ├── type: "text" or "image_description"
          ├── page number
          └── source filename