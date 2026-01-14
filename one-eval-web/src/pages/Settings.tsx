import { useMemo, useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Plus, Save, Database, Cpu, Cloud, KeyRound, Trash2, PlugZap } from "lucide-react";
import { motion } from "framer-motion";

interface ModelConfig {
  name: string;
  path: string;
  template_type: string;
}

export const Settings = () => {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [newModel, setNewModel] = useState({ name: "", path: "", template_type: "chatml" });
  const [loading, setLoading] = useState(false);
  const [apiBaseUrl, setApiBaseUrl] = useState(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000");
  const [savingApi, setSavingApi] = useState(false);
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

  const agentUrlPresets = useMemo(
    () => [
      { label: "Self-host (OpenAI Compatible) - /v1/chat/completions", value: "http://123.129.219.111:3000/v1/chat/completions" },
      { label: "Self-host (OpenAI Compatible) - /v1", value: "http://123.129.219.111:3000/v1" },
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

  useEffect(() => {
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
      setNewModel({ name: "", path: "", template_type: "chatml" });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleSaveApiConfig = async () => {
    setSavingApi(true);
    localStorage.setItem("oneEval.apiBaseUrl", apiBaseUrl);
    setSavingApi(false);
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
      setAgentApiKeyInput("");
      setAgentTestResult(null);
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
      const res = await axios.post(`${apiBaseUrl}/api/config/agent/test`);
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

  const handleTestConnection = async () => {
    try {
      await axios.get(`${apiBaseUrl}/health`);
      alert("Connection OK");
    } catch {
      alert("Connection failed");
    }
  };

  return (
    <div className="p-12 max-w-5xl mx-auto space-y-12">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Configuration</h1>
        <p className="text-muted-foreground text-lg">Manage your evaluation environment and model registry.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Agent Config */}
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
          <Card className="space-y-6">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    <Cpu className="w-6 h-6" />
                </div>
                <div>
                  <CardTitle>One-Eval Backend</CardTitle>
                  <CardDescription>Configure the One-Eval API server endpoint.</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>API Endpoint</Label>
                <Input value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} placeholder="http://localhost:8000" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Button className="w-full" variant="outline" onClick={handleTestConnection}>
                  Test Connection
                </Button>
                <Button
                  className="w-full text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                  onClick={handleSaveApiConfig}
                  disabled={savingApi}
                >
                  {savingApi ? "Saving..." : "Save"}
                </Button>
              </div>
            </CardContent>

            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-violet-500/10 text-violet-600">
                  <PlugZap className="w-6 h-6" />
                </div>
                <div>
                  <CardTitle>Agent Server</CardTitle>
                  <CardDescription>Choose provider URL, model, and API key.</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
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

              <div className="grid grid-cols-2 gap-3">
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
                  {agentApiKeySet && <span className="text-xs text-slate-500">Saved</span>}
                </div>
                <Input
                  type="password"
                  value={agentApiKeyInput}
                  onChange={(e) => setAgentApiKeyInput(e.target.value)}
                  placeholder="sk-... (won't be auto-filled)"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Button variant="outline" className="w-full" onClick={handleTestAgentConnection} disabled={testingAgent}>
                  {testingAgent ? "Testing..." : "Test Agent"}
                </Button>
                <Button
                  className="w-full text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                  onClick={handleSaveAgentConfig}
                  disabled={savingAgent}
                >
                  {savingAgent ? "Saving..." : "Save Agent"}
                </Button>
              </div>
              <div className="grid grid-cols-1 gap-2">
                <Button variant="outline" className="w-full" onClick={handleClearAgentApiKey} disabled={savingAgent}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear API Key
                </Button>
                {agentTestResult && (
                  <div className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-md px-3 py-2">
                    {agentTestResult}
                  </div>
                )}
              </div>
            </CardContent>

            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600">
                  <Cloud className="w-6 h-6" />
                </div>
                <div>
                  <CardTitle>HuggingFace</CardTitle>
                  <CardDescription>Mirror & token for datasets/models access.</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>HF Endpoint (Mirror)</Label>
                <Input
                  value={hfEndpoint}
                  onChange={(e) => setHfEndpoint(e.target.value)}
                  placeholder="https://hf-mirror.com"
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>HF Token</Label>
                  {hfTokenSet && <span className="text-xs text-slate-500">Saved</span>}
                </div>
                <Input
                  type="password"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_..."
                />
                <div className="text-xs text-slate-500">
                  Token is not auto-filled. Leave empty to keep current token.
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Button
                  className="w-full text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                  onClick={handleSaveHfConfig}
                  disabled={savingHf}
                >
                  {savingHf ? "Saving..." : <><KeyRound className="w-4 h-4 mr-2" /> Save HF</>}
                </Button>
                <Button
                  className="w-full"
                  variant="outline"
                  onClick={handleClearHfToken}
                  disabled={savingHf}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear Token
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Model Registry */}
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
        >
          <Card className="h-full flex flex-col">
             <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-500">
                    <Database className="w-6 h-6" />
                </div>
                <div>
                    <CardTitle>Target Model Registry</CardTitle>
                    <CardDescription>Register models to be evaluated.</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6 flex-1">
              {/* Add New */}
              <div className="p-4 border rounded-lg bg-secondary/20 space-y-4">
                <h4 className="text-sm font-medium flex items-center gap-2">
                    <Plus className="w-4 h-4" /> Add New Model
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Model Name</Label>
                    <Input 
                        placeholder="e.g. Qwen2.5-7B" 
                        value={newModel.name}
                        onChange={e => setNewModel({...newModel, name: e.target.value})}
                    />
                  </div>
                   <div className="space-y-2">
                    <Label>Template Type</Label>
                    <Input 
                        placeholder="chatml" 
                        value={newModel.template_type}
                        onChange={e => setNewModel({...newModel, template_type: e.target.value})}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                    <Label>Model Path / HuggingFace ID</Label>
                    <Input 
                        placeholder="/mnt/models/..." 
                        value={newModel.path}
                        onChange={e => setNewModel({...newModel, path: e.target.value})}
                    />
                </div>
                <Button onClick={handleSaveModel} disabled={loading} className="w-full">
                    {loading ? "Saving..." : <><Save className="w-4 h-4 mr-2"/> Save to Registry</>}
                </Button>
              </div>

              {/* List */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">Registered Models</h4>
                {models.length === 0 && <p className="text-sm text-muted-foreground italic">No models registered yet.</p>}
                <div className="max-h-[300px] overflow-y-auto space-y-2 pr-2">
                    {models.map((m, i) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-md border bg-card hover:bg-accent/50 transition-colors">
                            <div>
                                <div className="font-medium text-sm">{m.name}</div>
                                <div className="text-xs text-muted-foreground truncate max-w-[200px]" title={m.path}>{m.path}</div>
                            </div>
                            <div className="px-2 py-1 rounded-full bg-secondary text-[10px] font-mono">
                                {m.template_type}
                            </div>
                        </div>
                    ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};
