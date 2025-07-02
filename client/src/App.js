import { Routes, Route } from 'react-router-dom';
import './App.css';
import Navbar from './components/Navbar';
import Annotate from './components/Annotate';
import Inference from './components/Inference';
import About from './components/About';
import Contact from './components/Contact';
import Upload from './components/Upload';

const App = () => {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/annotate" element={<Annotate />} />
        <Route path="/inference" element={<Inference />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </div>
  );
}

export default App;
