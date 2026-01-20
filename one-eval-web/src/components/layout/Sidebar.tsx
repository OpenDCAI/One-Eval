import { Link, useLocation } from "react-router-dom";
import { Home, Play, Settings, Library, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

const SidebarItem = ({
  icon: Icon,
  label,
  to,
  active,
}: {
  icon: any;
  label: string;
  to: string;
  active: boolean;
}) => {
  return (
    <Link
      to={to}
      className={cn(
        "relative flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-200 group",
        active
          ? "bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-md shadow-blue-600/20"
          : "text-slate-500 hover:bg-slate-100 hover:text-slate-900"
      )}
    >
      <Icon className="w-5 h-5" />
      
      {/* Tooltip */}
      <span className="absolute left-14 px-2 py-1 bg-white text-slate-900 text-xs font-medium rounded opacity-0 -translate-x-2 pointer-events-none group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 shadow-md border border-slate-200 z-50 whitespace-nowrap">
        {label}
      </span>
    </Link>
  );
};

export const Sidebar = () => {
  const location = useLocation();

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-16 bg-white border-r border-slate-200 flex flex-col items-center py-6 z-[100]">
      {/* Logo */}
      <div className="mb-8">
        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-violet-600 rounded-xl flex items-center justify-center text-white shadow-md shadow-blue-600/20">
           <Layers className="w-6 h-6" />
        </div>
      </div>

      {/* Nav Items */}
      <nav className="flex flex-col gap-4 w-full items-center">
        <SidebarItem
          icon={Home}
          label="Home"
          to="/"
          active={location.pathname === "/"}
        />
        <SidebarItem
          icon={Play}
          label="Run Evaluation"
          to="/eval"
          active={location.pathname === "/eval"}
        />
        <SidebarItem
          icon={Library}
          label="Benchmark Gallery"
          to="/gallery"
          active={location.pathname === "/gallery"}
        />
        <SidebarItem
          icon={Settings}
          label="Settings"
          to="/settings"
          active={location.pathname === "/settings"}
        />
      </nav>

      {/* Footer Status */}
      <div className="mt-auto mb-4">
         <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" />
      </div>
    </aside>
  );
};
