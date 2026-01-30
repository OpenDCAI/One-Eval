import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowRight, Activity, Zap, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCallback } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";

export const Home = () => {
  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  return (
    <div className="h-screen w-full bg-white flex flex-col font-['Inter'] overflow-hidden relative">
      
      {/* Particles Background */}
      <div className="absolute inset-0 z-0">
      <Particles
        id="tsparticles"
        init={particlesInit}
        className="w-full h-full"
        options={{
          background: {
            color: {
              value: "transparent",
            },
          },
          fpsLimit: 120,
          interactivity: {
            events: {
              onClick: {
                enable: true,
                mode: "push",
              },
              onHover: {
                enable: true,
                mode: "grab",
              },
              resize: true,
            },
            modes: {
              push: {
                quantity: 4,
              },
              grab: {
                distance: 140,
                links: {
                  opacity: 0.5,
                  color: "#2563eb", // blue-600
                },
              },
            },
          },
          particles: {
            color: {
              value: "#94a3b8", // slate-400
            },
            links: {
              color: "#cbd5e1", // slate-300
              distance: 150,
              enable: true,
              opacity: 0.3,
              width: 1,
            },
            move: {
              direction: "none",
              enable: true,
              outModes: {
                default: "bounce",
              },
              random: false,
              speed: 1,
              straight: false,
            },
            number: {
              density: {
                enable: true,
                area: 800,
              },
              value: 80,
            },
            opacity: {
              value: 0.5,
            },
            shape: {
              type: "circle",
            },
            size: {
              value: { min: 1, max: 3 },
            },
          },
          detectRetina: true,
        }}
      />
      </div>

      {/* Subtle Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      {/* Navbar Placeholder */}
      <nav className="flex justify-between items-center px-8 py-6 z-10">
        <div className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-violet-600 rounded-lg flex items-center justify-center text-white shadow-md shadow-blue-600/20">
                <Layers className="w-5 h-5" />
            </div>
            One Eval
        </div>
        <div className="flex gap-4">
            <Button variant="ghost" className="text-slate-600 hover:text-slate-900">Documentation</Button>
            <Button variant="ghost" className="text-slate-600 hover:text-slate-900">GitHub</Button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center text-center px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl space-y-8"
        >
          {/* Badge */}
          <div className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-600 shadow-sm mb-4">
            <span className="flex h-2 w-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>
            v1.0 is now available
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 leading-[1.1]">
            One Eval <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-violet-600">
              evaluate in one via agents
            </span>
          </h1>
          
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Orchestrate complex evaluation workflows with a unified, graph-based engine. 
            From dataset discovery to granular metrics, all in one place.
          </p>

          <div className="flex gap-4 justify-center pt-4">
            <Link to="/eval">
              <Button size="lg" className="h-12 px-8 text-base text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-lg shadow-blue-600/20">
                Start Evaluating
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
            <Link to="/gallery">
              <Button size="lg" variant="outline" className="h-12 px-8 text-base border-slate-200 hover:bg-slate-50">
                View Gallery
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Feature Highlights */}
        <motion.div 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20 max-w-5xl w-full px-4"
        >
            {[
                { icon: Zap, title: "Instant Setup", desc: "Connect your model and start evaluating in seconds." },
                { icon: Layers, title: "Graph Engine", desc: "Powered by LangGraph for complex, stateful workflows." },
                { icon: Activity, title: "Deep Metrics", desc: "Get granular insights beyond just accuracy scores." }
            ].map((feature, i) => (
                <div key={i} className="flex flex-col items-center p-6 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                    <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 mb-4">
                        <feature.icon className="w-6 h-6" />
                    </div>
                    <h3 className="font-semibold text-slate-900 mb-2">{feature.title}</h3>
                    <p className="text-sm text-slate-500">{feature.desc}</p>
                </div>
            ))}
        </motion.div>
      </main>
    </div>
  );
};
