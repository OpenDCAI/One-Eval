import { useMemo, useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { 
  Plus, Save, Database, Cloud, KeyRound, Trash2, PlugZap, 
  ChevronDown, CheckCircle2
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ModelConfig {
  name: string;
  path: string;
}

interface SettingsCardProps {
  title: string;
  description: string;
  icon: React.ElementType;
  iconColorClass?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const SettingsCard = ({ 
  title, 
  description, 
  icon: Icon, 
  iconColorClass = "bg-primary/10 text-primary", 
  children, 
  defaultOpen = false 
}: SettingsCardProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className="overflow-hidden border-slate-200 shadow-sm hover:shadow-md transition-all duration-300">
      <CardHeader 
        className="cursor-pointer bg-slate-50/30 hover:bg-slate-50/80 transition-colors p-6"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-2.5 rounded-xl ${iconColorClass}`}>
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <CardTitle className="text-lg font-semibold text-slate-900">{title}</CardTitle>
              <CardDescription className="text-slate-500 mt-1">{description}</CardDescription>
            </div>
          </div>
          <ChevronDown 
            className={`w-5 h-5 text-slate-400 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`} 
          />
        </div>
      </CardHeader>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            <div className="border-t border-slate-100">
              <CardContent className="p-6 pt-6 space-y-6">
                {children}
              </CardContent>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};

export const Settings = () => {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [newModel, setNewModel] = useState({ name: "", path: "" });
  const [loading, setLoading] = useState(false);
  const [apiBaseUrl] = useState(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000");
  const [hfEndpoint, setHfEndpoint] = useState("https://hf-mirror.com");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [savingHf, setSavingHf] = useState(false);
  const [agentBaseUrl, setAgentBaseUrl] = useState("http://123.129.219.111:3000/v1");
  const [agentModel, setAgentModel] = useState("gpt-4o");
  const [agentApiKeyInput, setAgentApiKeyInput] = useState("");
  const [agentApiKeySet, setAgentApiKeySet] = useState(false);
  const [agentTimeoutS, setAgentTimeoutS] = useState(15);
  const [savingAgent, setSavingAgent] = useState(false);
  const [testingAgent, setTestingAgent] = useState(false);
  const [agentTestResult, setAgentTestResult] = useState<string | null>(null);
  const [showAgentSuccess, setShowAgentSuccess] = useState(false);

  const agentUrlPresets = useMemo(
    () => [
      { label: "yuchaAPI", value: "http://123.129.219.111:3000/v1/chat/completions" },
      { label: "OpenAI", value: "https://api.openai.com/v1" },
      { label: "OpenRouter", value: "https://openrouter.ai/api/v1" },
      { label: "Apiyi (OpenAI Compatible)", value: "https://api.apiyi.com/v1" },
      { label: "Custom...", value: "__custom__" },
    ],
    []
  );
  const agentUrlPresetValue = useMemo(() => {
    const hit = agentUrlPresets.find((p) => p.value === agentBaseUrl);
    return hit ? hit.value : "__custom__";
  }, [agentUrlPresets, agentBaseUrl]);

  const isValidHttpUrl = (u: string) => {
    try {
      const parsed = new URL(u);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  };

  useEffect(() => {
    if (!isValidHttpUrl(apiBaseUrl)) return;
    fetchModels();
    fetchHfConfig();
    fetchAgentConfig();
  }, [apiBaseUrl]);

  const fetchModels = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/models`);
      setModels(res.data);
    } catch (e) {
      console.error("Failed to fetch models", e);
    }
  };

  const handleSaveModel = async () => {
    if (!newModel.name || !newModel.path) return;
    setLoading(true);
    try {
      await axios.post(`${apiBaseUrl}/api/models`, newModel);
      setModels([...models, newModel]);
      setNewModel({ name: "", path: "" });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const fetchHfConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/hf`);
      setHfEndpoint(res.data.endpoint || "https://hf-mirror.com");
      setHfTokenSet(Boolean(res.data.token_set));
    } catch (e) {
      setHfEndpoint("https://hf-mirror.com");
      setHfTokenSet(false);
    }
  };

  const handleSaveHfConfig = async () => {
    setSavingHf(true);
    try {
      const payload: any = { endpoint: hfEndpoint };
      if (hfToken.trim()) payload.token = hfToken;
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, payload);
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const handleClearHfToken = async () => {
    setSavingHf(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, { clear_token: true });
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const fetchAgentConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/agent`);
      setAgentBaseUrl(res.data.base_url || "http://123.129.219.111:3000/v1");
      setAgentModel(res.data.model || "gpt-4o");
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || 15));
      setAgentApiKeyInput("");
    } catch (e) {
      setAgentBaseUrl("http://123.129.219.111:3000/v1");
      setAgentModel("gpt-4o");
      setAgentApiKeySet(false);
      setAgentTimeoutS(15);
      setAgentApiKeyInput("");
    }
  };

  const handleSaveAgentConfig = async () => {
    setSavingAgent(true);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      if (agentApiKeyInput.trim()) payload.api_key = agentApiKeyInput.trim();
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, payload);
      setAgentBaseUrl(res.data.base_url || agentBaseUrl);
      setAgentModel(res.data.model || agentModel);
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || agentTimeoutS));
      // Keep the input and result visible so user knows what happened
      // setAgentApiKeyInput(""); 
      // setAgentTestResult(null);
      setShowAgentSuccess(true);
      setTimeout(() => setShowAgentSuccess(false), 3000);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleClearAgentApiKey = async () => {
    setSavingAgent(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, { clear_api_key: true });
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentApiKeyInput("");
      setAgentTestResult(null);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleTestAgentConnection = async () => {
    setTestingAgent(true);
    setAgentTestResult(null);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      // Send the currently input API key if it's not empty, otherwise don't send it (let backend use saved key)
      // Actually, if we want to test "what I just typed", we should send it even if empty string?
      // But if user has a saved key and clears the input, maybe they mean "use saved"?
      // No, consistent UX: "Test" tests the *current form values*.
      // If user clears the input, they might mean "no auth".
      // However, for security, we don't auto-fill the input with the saved key.
      // So if input is empty, and there IS a saved key (agentApiKeySet is true), we probably want to use the saved key.
      // If input is NOT empty, use the input.
      if (agentApiKeyInput.trim()) {
        payload.api_key = agentApiKeyInput.trim();
      } else if (!agentApiKeySet) {
          // No saved key, and no input key -> send empty to override any potential default? 
          // Backend falls back to saved config if req.api_key is None.
          // If we send "", backend treats it as empty key.
          // If we don't send it, backend uses saved key.
          // If there is NO saved key (agentApiKeySet false), backend has None.
          // So if input is empty and not saved, we can just send nothing.
      } else {
         // Input empty, but key is saved. 
         // We should NOT send api_key field so backend uses the saved one.
      }

      const res = await axios.post(`${apiBaseUrl}/api/config/agent/test`, payload);
      if (res.data.ok) {
        setAgentTestResult(`OK (${res.data.mode})`);
      } else {
        const code = res.data.status_code ? ` [${res.data.status_code}]` : "";
        setAgentTestResult(`FAILED${code}: ${res.data.detail}`);
      }
    } catch (e) {
      setAgentTestResult("FAILED: request error");
    }
    setTestingAgent(false);
  };

  return (
    <div className="p-12 max-w-[1600px] mx-auto space-y-8">
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Settings</h1>
        <p className="text-slate-500 text-lg">Configure your evaluation environment and model registry.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-start">
        {/* 1. One-Eval Backend (Hidden) */}

        {/* 2. Agent Server */}
        <SettingsCard
          title="Agent Server"
          description="Configure the LLM provider (e.g. OpenAI, vLLM, etc.) used for evaluation."
          icon={PlugZap}
          iconColorClass="bg-violet-500/10 text-violet-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>Provider URL</Label>
              <div className="grid grid-cols-1 gap-2">
                <select
                  value={agentUrlPresetValue}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v !== "__custom__") setAgentBaseUrl(v);
                  }}
                  className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {agentUrlPresets.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
                <Input
                  value={agentBaseUrl}
                  onChange={(e) => setAgentBaseUrl(e.target.value)}
                  placeholder="https://.../v1  or  https://.../v1/chat/completions"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Model</Label>
                <select
                  value={agentModel}
                  onChange={(e) => setAgentModel(e.target.value)}
                  className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  <option value="gpt-4o">gpt-4o</option>
                  <option value="gpt-5.1">gpt-5.1</option>
                  <option value="gpt-5.2">gpt-5.2</option>
                  <option value="deepseek-v3">deepseek-v3</option>
                  <option value="deepseek-r1">deepseek-r1</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label>Timeout (s)</Label>
                <Input
                  type="number"
                  value={agentTimeoutS}
                  onChange={(e) => setAgentTimeoutS(Number(e.target.value || 15))}
                  className="border-slate-200"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>API Key</Label>
                {agentApiKeySet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">Key Saved</span>}
              </div>
              <Input
                type="password"
                value={agentApiKeyInput}
                onChange={(e) => setAgentApiKeyInput(e.target.value)}
                placeholder="sk-... (won't be auto-filled for security)"
              />
            </div>

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={handleTestAgentConnection} disabled={testingAgent}>
                {testingAgent ? "Testing..." : "Test Connection"}
              </Button>
              <Button
                className={`flex-1 text-white transition-all duration-300 ${
                  showAgentSuccess 
                    ? "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/20" 
                    : "bg-slate-900 hover:bg-slate-800"
                }`}
                onClick={handleSaveAgentConfig}
                disabled={savingAgent}
              >
                {savingAgent ? (
                  "Saving..."
                ) : showAgentSuccess ? (
                  <><CheckCircle2 className="w-4 h-4 mr-2" /> Saved!</>
                ) : (
                  "Save Configuration"
                )}
              </Button>
            </div>

            <div className="flex items-center justify-between pt-2">
               <Button variant="ghost" size="sm" className="text-red-500 hover:text-red-600 hover:bg-red-50" onClick={handleClearAgentApiKey} disabled={savingAgent}>
                <Trash2 className="w-4 h-4 mr-2" />
                Clear API Key
              </Button>
              {agentTestResult && (
                <div className={`text-xs px-3 py-1.5 rounded-md font-mono ${agentTestResult.startsWith("OK") ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-red-50 text-red-700 border border-red-200"}`}>
                  {agentTestResult}
                </div>
              )}
            </div>
          </div>
        </SettingsCard>

        {/* 3. HuggingFace */}
        <SettingsCard
          title="HuggingFace Configuration"
          description="Configure HF mirror endpoint and access token for downloading models/datasets."
          icon={Cloud}
          iconColorClass="bg-amber-500/10 text-amber-600"
          defaultOpen={false}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>HF Mirror Endpoint</Label>
              <Input
                value={hfEndpoint}
                onChange={(e) => setHfEndpoint(e.target.value)}
                placeholder="https://hf-mirror.com"
              />
              <p className="text-xs text-slate-500">Default: https://hf-mirror.com</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>HF Token</Label>
                {hfTokenSet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">Token Saved</span>}
              </div>
              <Input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_..."
              />
              <p className="text-xs text-slate-500">
                Leave empty to keep the currently saved token.
              </p>
            </div>

            <div className="flex gap-3">
              <Button
                className="flex-1 text-white bg-slate-900 hover:bg-slate-800"
                onClick={handleSaveHfConfig}
                disabled={savingHf}
              >
                {savingHf ? "Saving..." : <><KeyRound className="w-4 h-4 mr-2" /> Save HF Config</>}
              </Button>
              <Button
                variant="outline"
                className="flex-1 text-red-500 hover:text-red-600 hover:bg-red-50"
                onClick={handleClearHfToken}
                disabled={savingHf}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear Token
              </Button>
            </div>
          </div>
        </SettingsCard>

        {/* 4. Model Registry */}
        <SettingsCard
          title="Target Model Registry"
          description="Register local or remote models that you want to evaluate."
          icon={Database}
          iconColorClass="bg-pink-500/10 text-pink-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            {/* Add New */}
            <div className="p-5 border border-slate-200 rounded-xl bg-slate-50/50 space-y-4">
              <h4 className="text-sm font-semibold flex items-center gap-2 text-slate-800">
                <Plus className="w-4 h-4" /> Add New Model
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Model Name</Label>
                  <Input 
                    placeholder="e.g. Qwen2.5-7B-Instruct" 
                    value={newModel.name}
                    onChange={e => setNewModel({...newModel, name: e.target.value})}
                    className="bg-white"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Model Path / HuggingFace ID</Label>
                  <Input 
                    placeholder="/mnt/models/..." 
                    value={newModel.path}
                    onChange={e => setNewModel({...newModel, path: e.target.value})}
                    className="bg-white"
                  />
                </div>
              </div>
              <Button onClick={handleSaveModel} disabled={loading} className="w-full text-white bg-slate-900 hover:bg-slate-800">
                {loading ? "Saving..." : <><Save className="w-4 h-4 mr-2"/> Add to Registry</>}
              </Button>
            </div>

            {/* List */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider text-xs">Registered Models</h4>
              {models.length === 0 && (
                <div className="text-center py-8 border-2 border-dashed border-slate-200 rounded-xl">
                  <p className="text-sm text-slate-400">No models registered yet.</p>
                </div>
              )}
              <div className="grid grid-cols-1 gap-3">
                {models.map((m, i) => (
                  <div key={i} className="flex items-center justify-between p-4 rounded-xl border border-slate-100 bg-white hover:border-slate-200 hover:shadow-sm transition-all">
                    <div className="flex-1 min-w-0 mr-4">
                      <div className="font-semibold text-slate-900">{m.name}</div>
                      <div className="text-xs text-slate-500 truncate font-mono mt-1" title={m.path}>{m.path}</div>
                    </div>
                    {/* Actions if needed, maybe delete later */}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </SettingsCard>

      </div>
    </div>
  );
};

export default Settings;
