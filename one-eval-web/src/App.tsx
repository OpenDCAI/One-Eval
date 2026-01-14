import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Home } from "@/pages/Home";
import { Eval } from "@/pages/Eval";
import { Settings } from "@/pages/Settings";
import { Gallery } from "@/pages/Gallery";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Landing Page without Layout */}
        <Route path="/" element={<Home />} />
        
        {/* Main App Layout */}
        <Route element={<Layout />}>
          <Route path="/eval" element={<Eval />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/gallery" element={<Gallery />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
