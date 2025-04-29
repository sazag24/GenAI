# Python Developer's Guide to LLMs Book Viewer

This is a simple web-based viewer for "The Python Developer's Guide to Building Intelligent Assistants with LLMs" book.

## Features

- View all chapters and research notes
- Simple navigation between chapters
- Toggle research notes visibility
- Responsive design for different screen sizes

## Getting Started

### Prerequisites

- Python 3.6 or higher

### Running the Application

1. Start the custom HTTP server with CORS support:

```bash
python server.py
```

2. Open your browser and navigate to:

```
http://localhost:8000/simple.html
```

This simpler version displays the book content correctly with proper markdown rendering.

You can also view the more complex version with additional features:

```
http://localhost:8000/index.html
```

## Troubleshooting

If you're having issues with CORS (Cross-Origin Resource Sharing) errors:

1. Make sure you're accessing the files through the HTTP server and not directly from the file system
2. Check that the server is running on port 8000
3. Clear your browser cache if you're still experiencing issues

## Structure

- `/docs/` - Contains all book chapters and research notes in markdown format
- `simple.html` - Basic version of the book viewer 
- `index.html` - Full-featured book viewer
- `server.py` - Custom HTTP server with CORS support

## Browser Support

The viewer has been tested on:
- Chrome
- Firefox
- Edge

## Notes

The viewer uses zero-md web component to render markdown files into HTML. This requires loading content through an HTTP server due to CORS restrictions in modern browsers.
