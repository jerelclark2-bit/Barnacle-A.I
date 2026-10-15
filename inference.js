/* BarnacleAI: real PyTorch CNN weights evaluated locally in the browser with TensorFlow.js.
   Convolution weights are transposed from PyTorch OIHW to TensorFlow HWIO.
   No photo is sent to a server. The trained threshold is 0.75; these are
   experimental candidate centers, not verified barnacle counts. */
(function(){
  'use strict';
  const TILE=128;
  const LIBRARIES=[
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    'https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js'
  ];
  let ready=null,weights=null,trainedThreshold=0.75;
  function loadScript(url){
    return new Promise((resolve,reject)=>{
      const s=document.createElement('script');s.src=url;s.async=true;
      s.onload=resolve;s.onerror=()=>{s.remove();reject(new Error('Could not load '+url))};
      document.head.appendChild(s);
    });
  }
  async function initialize(){
    if(typeof window.tf==='undefined'){
      let loaded=false;
      for(const url of LIBRARIES){
        try{await loadScript(url);if(window.tf){loaded=true;break}}catch(e){console.warn(e.message)}
      }
      if(!loaded)throw new Error('AI engine could not load. Check your internet connection, then retry.');
    }
    await tf.ready();
    // Prefer accelerated WebGL where supported; TensorFlow.js can also use CPU.
    try{if(tf.findBackend('webgl')){await tf.setBackend('webgl');await tf.ready()}}catch(e){console.warn('Using fallback inference backend',e)}
    const response=await fetch('./model_weights.json?v=20261015cnn',{cache:'force-cache'});
    if(!response.ok)throw new Error('CNN weights are unavailable (HTTP '+response.status+').');
    const saved=await response.json();
    if(!saved.weights || !saved.weights['enc1.0.weight'] || !saved.weights['out.bias'])throw new Error('CNN weight file is incomplete.');
    trainedThreshold=Number(saved.threshold)||0.75;
    const tensors={};
    for(const [key,entry] of Object.entries(saved.weights)){
      const binary=atob(entry.b64),bytes=new Uint8Array(binary.length);
      for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
      tensors[key]=tf.tensor(new Float32Array(bytes.buffer),entry.shape,'float32');
    }
    weights=tensors;
    return {threshold:trainedThreshold,backend:tf.getBackend(),epochs:saved.trained_epochs};
  }
  async function load(){
    if(!ready)ready=initialize().catch(e=>{ready=null;throw e});
    return ready;
  }
  function convolution(x,name,activation=true){
    const y=tf.add(tf.conv2d(x,weights[name+'.weight'],1,'same'),weights[name+'.bias']);
    return activation?tf.relu(y):y;
  }
  function block(x,name){
    return convolution(convolution(x,name+'.0'),name+'.2');
  }
  function forward(x){
    return tf.tidy(()=>{
      const s1=block(x,'enc1');
      const s2=block(tf.maxPool(s1,2,2,'valid'),'enc2');
      const z=block(tf.maxPool(s2,2,2,'valid'),'bottleneck');
      // Half-pixel resize matches torch.nn.functional.interpolate(align_corners=False).
      const u2=block(tf.concat([tf.image.resizeBilinear(z,[s2.shape[1],s2.shape[2]],false,true),s2],3),'up2');
      const u1=block(tf.concat([tf.image.resizeBilinear(u2,[s1.shape[1],s1.shape[2]],false,true),s1],3),'up1');
      return tf.sigmoid(convolution(u1,'out',false));
    });
  }
  function reflect(v,n){
    if(n<=1)return 0;
    while(v<0||v>=n){if(v<0)v=-v;else v=2*n-2-v}
    return v;
  }
  async function probabilityMap(rgba,width,height,onProgress){
    await load();
    if(width<16||height<16)throw new Error('Image is too small for reliable processing.');
    const nx=Math.ceil(width/TILE),ny=Math.ceil(height/TILE),total=nx*ny;
    const probabilities=new Float32Array(width*height);
    let done=0;
    for(let ty=0;ty<ny;ty++)for(let tx=0;tx<nx;tx++){
      const input=new Float32Array(TILE*TILE*3);
      for(let y=0;y<TILE;y++){
        const iy=reflect(ty*TILE+y,height);
        for(let x=0;x<TILE;x++){
          const ix=reflect(tx*TILE+x,width),src=4*(iy*width+ix),dst=3*(y*TILE+x);
          input[dst]=rgba[src]/255;input[dst+1]=rgba[src+1]/255;input[dst+2]=rgba[src+2]/255;
        }
      }
      const x=tf.tensor4d(input,[1,TILE,TILE,3]);
      let out;
      try{
        out=forward(x);
        const result=await out.data();
        for(let y=0;y<TILE;y++){
          const iy=ty*TILE+y;if(iy>=height)break;
          for(let px=0;px<TILE;px++){
            const ix=tx*TILE+px;if(ix>=width)break;
            probabilities[iy*width+ix]=result[y*TILE+px];
          }
        }
      }finally{x.dispose();if(out)out.dispose()}
      done++;
      if(onProgress)onProgress(done,total);
      if(done%3===0)await tf.nextFrame();
    }
    return probabilities;
  }
  function findPeaks(prob,width,height,minThreshold=0.65,radius=7){
    const points=[];
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const value=prob[y*width+x];if(value<minThreshold)continue;
      let maximum=true;
      for(let dy=-radius;dy<=radius&&maximum;dy++){
        const yy=y+dy;if(yy<0||yy>=height)continue;
        for(let dx=-radius;dx<=radius;dx++){
          const xx=x+dx;if(xx<0||xx>=width||(dx===0&&dy===0))continue;
          const other=prob[yy*width+xx];
          if(other>value||(other===value&&(yy<y||(yy===y&&xx<x)))){maximum=false;break}
        }
      }
      if(maximum)points.push({x,y,score:value});
    }
    return points;
  }
  window.BarnacleCNN={load,probabilityMap,findPeaks,get threshold(){return trainedThreshold}};
})();