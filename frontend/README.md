# RAG AI Assistant - Frontend

Beautiful, modern React frontend for the RAG AI Voice Assistant.

## 🎨 Features

- **Modern UI/UX**: Clean, responsive design with TailwindCSS
- **Dark Mode**: Toggle between light and dark themes
- **Real-time Chat**: Smooth message animations and typing indicators
- **Responsive**: Works on desktop, tablet, and mobile devices
- **Error Handling**: User-friendly error messages
- **Quick Actions**: Suggested queries for easy interaction

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # App header with dark mode toggle
│   │   ├── ChatWindow.jsx      # Main chat interface
│   │   ├── MessageBubble.jsx   # Individual message component
│   │   ├── TypingIndicator.jsx # Loading animation
│   │   └── InputBox.jsx        # Message input field
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles with Tailwind
├── index.html                  # HTML template
├── package.json                # Dependencies
├── vite.config.js              # Vite configuration
└── tailwind.config.js          # Tailwind configuration
```

## 🎨 UI Components

### Header
- Logo and branding
- Dark mode toggle
- Responsive design

### ChatWindow
- Welcome screen with quick actions
- Message history with smooth scrolling
- Typing indicator
- Auto-scroll to latest message

### MessageBubble
- User messages (blue, right-aligned)
- Assistant messages (white/gray, left-aligned)
- Avatar icons
- Smooth animations

### InputBox
- Multi-line text input
- Send button with loading state
- Enter to send, Shift+Enter for new line
- Disabled state during loading

## 🎯 API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`.

### Endpoints Used

- `POST /chat` - Send message and get response

### Example Request

```javascript
axios.post('http://localhost:8000/chat', {
  message: "What programs do you offer?"
})
```

### Example Response

```json
{
  "response": "We offer 6 academic programs...",
  "success": true,
  "error": null
}
```

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to change the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        500: '#0ea5e9',  // Change this
        // ...
      },
    },
  },
}
```

### Animations

Custom animations are defined in `index.css`:

- `fade-in` - Fade in animation
- `slide-up` - Slide up animation
- `typing-dot` - Typing indicator animation

## 🐛 Troubleshooting

### "Failed to get response"

**Cause**: Backend API is not running or not accessible.

**Solution**: 
1. Make sure the API server is running: `python api.py`
2. Check if it's accessible at `http://localhost:8000`
3. Verify CORS is enabled in the backend

### Styles not loading

**Cause**: Tailwind CSS not processing correctly.

**Solution**:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Restart the dev server

### Port 3000 already in use

**Solution**: Change the port in `vite.config.js`:

```javascript
server: {
  port: 3001,  // Change this
}
```

## 📝 Development Notes

- The app uses Vite for fast development and building
- TailwindCSS provides utility-first styling
- Lucide React provides modern icons
- Axios handles API requests
- React hooks manage state and side effects

## 🚀 Deployment

For production deployment:

1. Build the app: `npm run build`
2. Deploy the `dist` folder to any static hosting service:
   - Netlify
   - Vercel
   - GitHub Pages
   - AWS S3 + CloudFront

## 📄 License

MIT License
