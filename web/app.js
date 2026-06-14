const FEATURE_FALLBACK_LIB = [
  {name:"Aspirin", smiles:"CC(=O)Oc1ccccc1C(=O)O"},
  {name:"Caffeine", smiles:"Cn1cnc2c1c(=O)n(C)c(=O)n2C"},
];

const App = {
  RDKit:null, session:null, art:null, feat:null, inName:null,
  score:{ok:0,n:0}, current:null, _seed:7,

  async init(){
    try{
      this.art = await (await fetch("./artifacts.json")).json();
      this.feat = this.art.features;

      setLoad("Loading RDKit (chemistry engine)...");
      this.RDKit = await window.initRDKitModule({
        locateFile: () => "https://unpkg.com/@rdkit/rdkit@2025.3.4-1.0.0/dist/RDKit_minimal.wasm"
      });

      setLoad("Loading the solubility model...");
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
      const buf = await (await fetch("./model.onnx")).arrayBuffer();
      this.session = await ort.InferenceSession.create(new Uint8Array(buf));
      this.inName = this.session.inputNames[0];

      this.fillStatic();
      this.bind();
      Tutor.init(this);
      document.getElementById("loading").style.opacity="0";
      setTimeout(()=>document.getElementById("loading").classList.add("hidden"),420);

      this.predict("CC(=O)Oc1ccccc1C(=O)O"); // aspirin, so the page is never empty
    }catch(e){
      setLoad("Failed to load: "+(e&&e.message?e.message:e));
      console.error(e);
    }
  },

  fillStatic(){
    const m = this.art.metrics, gbm = m[this.art.best_model];
    document.getElementById("heroR2").textContent = gbm.r2.toFixed(2);
    document.getElementById("mR2").textContent = gbm.r2.toFixed(3);
    document.getElementById("mRMSE").textContent = gbm.rmse.toFixed(2);
    document.getElementById("mN").textContent = this.art.dataset.n.toLocaleString();
    document.getElementById("mModel").textContent = this.art.best_model.replace(/_/g," ");
    document.getElementById("mGbm").textContent = gbm.rmse.toFixed(2);
    if(m.linear) document.getElementById("mLin").textContent = m.linear.rmse.toFixed(2);
    document.getElementById("cite").innerHTML =
      "Dataset: <b>"+this.art.dataset.name+"</b> ("+this.art.dataset.n+" compounds). "+
      this.art.dataset.citation+" Distributed via MoleculeNet (Wu et al., <i>Chem. Sci.</i> 2018).";
    const chips = document.getElementById("chips");
    (this.art.examples||FEATURE_FALLBACK_LIB).forEach(ex=>{
      const b=document.createElement("button");
      b.className="chip"; b.textContent=ex.name; b.dataset.smiles=ex.smiles; b.title=ex.smiles;
      b.onclick=()=>{document.getElementById("smiles").value=ex.smiles; this.predict(ex.smiles);};
      chips.appendChild(b);
    });
    document.getElementById("repolink").href = "https://github.com/garyzhang1006/solu";
  },

  bind(){
    document.getElementById("go").onclick=()=>this.predict(document.getElementById("smiles").value.trim());
    document.getElementById("smiles").addEventListener("keydown",e=>{if(e.key==="Enter")this.predict(e.target.value.trim());});
    document.getElementById("rnd").onclick=()=>{
      const ex=this.art.examples[Math.floor(this.fakeRandom()*this.art.examples.length)];
      document.getElementById("smiles").value=ex.smiles; this.predict(ex.smiles);
    };
    document.querySelectorAll(".quizbtns [data-guess]").forEach(b=>{ b.onclick=()=>this.quiz(b.dataset.guess); });
  },

  fakeRandom(){ this._seed=(this._seed*9301+49297)%233280; return this._seed/233280; },

  descriptors(mol){
    const d = JSON.parse(mol.get_descriptors());
    return this.art.feature_order.map((k,i)=> d[this.art.js_keys[i]]);
  },

  async predict(smiles){
    const err=document.getElementById("err"); err.textContent="";
    if(!smiles){err.textContent="Enter a SMILES string or pick an example.";return;}
    let mol;
    try{ mol = this.RDKit.get_mol(smiles); }catch(e){ mol=null; }
    if(!mol || !mol.is_valid()){
      if(mol) mol.delete();
      err.textContent="Could not parse that SMILES. Try an example chip below.";
      return;
    }

    const svg = mol.get_svg(420,240);
    document.getElementById("structure").innerHTML = svg;
    let canon = smiles;
    try{ canon = mol.get_smiles(); }catch(e){}
    document.getElementById("iupac").textContent = "Canonical SMILES: "+canon;

    const x = this.descriptors(mol);
    mol.delete();
    const tensor = new ort.Tensor("float32", Float32Array.from(x), [1, x.length]);
    const out = await this.session.run({ [this.inName]: tensor });
    const logS = out[this.session.outputNames[0]].data[0];

    this.current = {smiles:canon, x, logS, name:this.nameFor(canon)};
    this.renderPrediction(this.current);
    this.renderContributions(x);
    this.renderTable(x);
    this.resetQuiz();
    Tutor.onPredict(this.current);
  },

  nameFor(canon){
    const hit=(this.art.examples||[]).find(e=>e.smiles===canon);
    return hit?hit.name:"custom";
  },

  classify(logS){
    if(logS> -1)  return {t:"Highly soluble", c:"var(--good)"};
    if(logS> -2)  return {t:"Soluble", c:"var(--good)"};
    if(logS> -3)  return {t:"Moderately soluble", c:"var(--accent2)"};
    if(logS> -4)  return {t:"Slightly soluble", c:"var(--warn)"};
    return {t:"Poorly soluble", c:"var(--bad)"};
  },

  fmtSolubility(logS, mw){
    const molL = Math.pow(10, logS);
    const gL = molL * mw;
    const mgL = gL*1000;
    if(mgL>=1e6) return (gL/1000).toFixed(1)+" kg/L";
    if(mgL>=1000) return gL.toFixed(gL>=10?0:1)+" g/L";
    if(mgL>=1) return mgL.toFixed(mgL>=10?0:1)+" mg/L";
    if(mgL>=1e-3) return (mgL*1000).toFixed(1)+" µg/L";
    return mgL.toExponential(1)+" mg/L";
  },

  renderPrediction(cur){
    const cls=this.classify(cur.logS);
    const v=document.getElementById("verdict");
    v.textContent=cls.t; v.style.color=cls.c;
    document.getElementById("logs").textContent=cur.logS.toFixed(2);
    const mw=cur.x[this.art.feature_order.indexOf("amw")];
    document.getElementById("mgl").textContent=this.fmtSolubility(cur.logS,mw);
    document.getElementById("molname").textContent=cur.name;
    const ys=this.art.target_stats;
    const lo=Math.min(ys.min,-9), hi=Math.max(ys.max,1.5);
    const pct=Math.max(2,Math.min(98,(cur.logS-lo)/(hi-lo)*100));
    document.getElementById("gmark").style.left=pct+"%";
    const rmse=this.art.metrics[this.art.best_model].rmse;
    document.getElementById("errband").innerHTML=
      "Typical model error ±"+rmse.toFixed(2)+" log units (about a "+Math.round(Math.pow(10,rmse))+"&times; range). Educational estimate.";
  },

  renderContributions(x){
    const {mean,scale}=this.art.scaler, coef=this.art.linear.coef;
    const items=this.art.feature_order.map((k,i)=>{
      const z=(x[i]-mean[i])/scale[i];
      return {key:k, label:this.feat[i].label, note:this.feat[i].note, c:coef[i]*z};
    }).sort((a,b)=>Math.abs(b.c)-Math.abs(a.c));
    const maxAbs=Math.max(...items.map(i=>Math.abs(i.c)),0.001);
    const wrap=document.getElementById("bars"); wrap.innerHTML="";
    items.forEach(it=>{
      const row=document.createElement("div"); row.className="bar";
      const pos=it.c>=0;
      const half=Math.abs(it.c)/maxAbs*50;
      const color=pos?"var(--good)":"var(--warn)";
      row.innerHTML=
        '<div class="lab" title="'+it.note.replace(/"/g,"&quot;")+'">'+it.label+'</div>'+
        '<div class="barwrap"><div class="barmid"></div>'+
          '<div class="barfill" style="background:'+color+';'+
          (pos?('left:50%;width:'+half+'%'):('left:'+(50-half)+'%;width:'+half+'%'))+'"></div></div>'+
        '<div class="barval">'+(pos?"+":"")+it.c.toFixed(2)+'</div>';
      wrap.appendChild(row);
    });
  },

  renderTable(x){
    const tb=document.querySelector("#desctbl tbody"); tb.innerHTML="";
    this.art.feature_order.forEach((k,i)=>{
      const f=this.feat[i], st=this.art.feature_stats[k];
      const val=x[i];
      const pct=Math.max(2,Math.min(100,(val-st.min)/((st.max-st.min)||1)*100));
      const unit=f.unit?(" "+f.unit):"";
      const tr=document.createElement("tr");
      tr.innerHTML="<td title='"+f.note.replace(/'/g,"")+"'>"+f.label+"</td>"+
        "<td class='v'>"+(Number.isInteger(val)?val:val.toFixed(2))+unit+"</td>"+
        "<td><div class='spark'><div class='sparkfill' style='width:"+pct+"%'></div></div></td>";
      tb.appendChild(tr);
    });
  },

  resetQuiz(){
    document.getElementById("quizq").textContent="Will "+(this.current?(this.current.name==="custom"?"this molecule":this.current.name):"it")+" dissolve well in water?";
    document.getElementById("quizres").textContent="";
  },
  quiz(guess){
    if(!this.current){document.getElementById("quizres").textContent="Pick a molecule first.";return;}
    const soluble=this.current.logS> -2;
    const correct=(guess==="sol")===soluble;
    this.score.n++; if(correct)this.score.ok++;
    const res=document.getElementById("quizres");
    res.style.color=correct?"var(--good2)":"var(--bad)";
    res.textContent=(correct?"✅ Right! ":"❌ Not quite. ")+
      this.current.name+" is "+(soluble?"fairly soluble":"poorly soluble")+
      " (log S "+this.current.logS.toFixed(2)+").";
    document.getElementById("score").textContent="Score: "+this.score.ok+" / "+this.score.n;
  }
};

// AI tutor: streams a grounded Claude Sonnet 4.6 explanation from tutor/server.py.
// Picks the structurally nearest REAL measured molecules (grounding.json) as the
// model's evidence, in the model's own scaler-standardized feature space.
// All model/dataset text reaches the DOM via textContent only (no innerHTML) -> XSS-safe.
const Tutor = {
  app:null, grounding:null, scale:null, idx:null, busy:false, cur:null,

  async init(app){
    this.app = app;
    const a = app.art;
    this.scale = a.feature_order.map((k,i)=> a.scaler.scale[i]);   // per-feature std
    const q   = document.getElementById("tutorQ");
    document.getElementById("tutorAsk").onclick     = ()=> this.ask(q.value.trim());
    document.getElementById("tutorExplain").onclick = ()=> this.ask("");
    q.addEventListener("keydown", e=>{ if(e.key==="Enter") this.ask(q.value.trim()); });
    this.setEnabled(false);
    try{
      const g = await (await fetch("./grounding.json")).json();
      this.grounding = g.molecules || [];
      const gOrder = g.feature_order || a.feature_order;
      this.idx = a.feature_order.map(k=> gOrder.indexOf(k));        // align corpus -> app order
      document.getElementById("tutorN").textContent = (g.n||this.grounding.length).toLocaleString();
    }catch(e){
      this.grounding = [];
      document.getElementById("tutorN").textContent = "the";
    }
  },

  setEnabled(on){
    document.getElementById("tutorAsk").disabled     = !on;
    document.getElementById("tutorExplain").disabled = !on;
  },

  onPredict(cur){
    this.cur = cur;
    this.setEnabled(true);
    document.getElementById("tutorOut").textContent = "";
    document.getElementById("tutorCites").replaceChildren();
    const note = document.getElementById("tutorNote");
    note.textContent = ""; note.className = "tutornote";
  },

  // k nearest real molecules by standardized-descriptor (Euclidean) distance.
  neighbors(x, k){
    if(!this.grounding || !this.grounding.length) return [];
    const scored = [];
    for(const g of this.grounding){
      if(g.smiles === this.cur.smiles) continue;   // never cite the molecule itself
      let d = 0;
      for(let i=0;i<x.length;i++){
        const diff = (x[i]-g.x[this.idx[i]])/(this.scale[i]||1);
        d += diff*diff;
      }
      scored.push([d,g]);
    }
    scored.sort((a,b)=>a[0]-b[0]);
    return scored.slice(0,k).map(s=>({smiles:s[1].smiles, logS:s[1].logS, source:s[1].source}));
  },

  descriptors(){
    const a = this.app.art, x = this.cur.x;
    return a.feature_order.map((k,i)=>{
      const f = a.feat[i], st = a.feature_stats[k];
      const pct = Math.max(1,Math.min(99, Math.round((x[i]-st.min)/((st.max-st.min)||1)*100)));
      return { label:f.label, value:(Number.isInteger(x[i])?x[i]:+x[i].toFixed(2)),
               unit:f.unit?(" "+f.unit):"", percentile:pct };
    });
  },

  renderCites(neighbors){
    const wrap = document.getElementById("tutorCites");
    wrap.replaceChildren();
    if(!neighbors.length) return;
    const lab = document.createElement("span");
    lab.className = "tutorcite"; lab.style.border="none"; lab.style.background="none";
    lab.textContent = "Cited measurements:";
    wrap.appendChild(lab);
    neighbors.forEach(n=>{
      const el = document.createElement("span");
      el.className = "tutorcite";
      el.textContent = "logS "+n.logS+" · "+n.source;
      el.title = n.smiles;
      wrap.appendChild(el);
    });
  },

  // Build the note from text + <code> segments without innerHTML.
  note(segs, cls){
    const note = document.getElementById("tutorNote");
    note.className = "tutornote"+(cls?(" "+cls):"");
    note.replaceChildren();
    segs.forEach(s=>{
      if(typeof s === "string"){ note.appendChild(document.createTextNode(s)); }
      else { const c = document.createElement("code"); c.textContent = s.code; note.appendChild(c); }
    });
  },

  async ask(question){
    if(this.busy || !this.cur) return;
    this.busy = true; this.setEnabled(false);
    const out  = document.getElementById("tutorOut");
    this.note([]);                                  // clear note

    const neighbors = this.neighbors(this.cur.x, 6);
    this.renderCites(neighbors);

    // streamed text node + blinking cursor (no innerHTML)
    out.replaceChildren();
    const textEl = document.createElement("span");
    const cursor = document.createElement("span"); cursor.className = "cursor";
    out.appendChild(textEl); out.appendChild(cursor);

    const mw = this.cur.x[this.app.art.feature_order.indexOf("amw")];
    const payload = {
      smiles: this.cur.smiles,
      name:   this.cur.name==="custom" ? null : this.cur.name,
      logS:   +this.cur.logS.toFixed(2),
      klass:  this.app.classify(this.cur.logS).t,
      mgl:    this.app.fmtSolubility(this.cur.logS, mw),
      descriptors: this.descriptors(),
      neighbors: neighbors,
      question: question || ""
    };

    let acc = "";
    try{
      const res = await fetch("/api/tutor", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      if(!res.ok || !res.body) throw new Error("no stream");
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for(;;){
        const {value, done} = await reader.read();
        if(done) break;
        buf += dec.decode(value, {stream:true});
        let nl;
        while((nl = buf.indexOf("\n\n")) >= 0){
          const line = buf.slice(0, nl); buf = buf.slice(nl+2);
          if(!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5).trim());
          if(ev.error) throw new Error(ev.error);
          if(ev.text){ acc += ev.text; textEl.textContent = acc; }
        }
      }
      cursor.remove();
      if(!acc.trim()) this.note(["The tutor returned an empty response."], "err");
    }catch(e){
      cursor.remove();
      const msg = (e&&e.message)||String(e);
      if(msg.includes("Failed to fetch")||msg.includes("NetworkError")||msg==="no stream")
        this.note(["Tutor server not reachable. Run ", {code:"export ANTHROPIC_API_KEY=sk-ant-..."},
                   " then ", {code:"python tutor/server.py"}, " and open ",
                   {code:"http://localhost:8000"}, "."], "err");
      else
        this.note(["Tutor error: "+msg], "err");
    }finally{
      this.busy = false; this.setEnabled(true);
    }
  }
};

function setLoad(m){const el=document.getElementById("loadmsg"); if(el)el.textContent=m;}
window.addEventListener("load", ()=>App.init());
