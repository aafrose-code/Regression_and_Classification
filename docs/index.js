async function runExample1() {
  const ids = [
    'box0c1', 'box1c1', 'box2c1', 'box3c1',
    'box4c1', 'box5c1', 'box6c1', 'box7c1'
  ];

  const x = new Float32Array(8);
  const predictions = document.getElementById('predictions1');

  for (let i = 0; i < ids.length; i++) {
    const field = document.getElementById(ids[i]);
    if (!field || field.value.trim() === '' || !Number.isFinite(Number(field.value))) {
      predictions.textContent = 'Please enter a valid number for every housing feature.';
      return;
    }
    x[i] = Number(field.value);
  }

  const tensorX = new ort.Tensor('float32', x, [1, 8]);

  try {
    predictions.textContent = 'Running prediction...';
    const session = await ort.InferenceSession.create('./regression.onnx');
    const results = await session.run({ input1: tensorX });
    const standardizedPrediction = results.output1.data[0];
    const prediction = standardizedPrediction * 115622.625 + 207194.703125;

    predictions.innerHTML = `
      <table>
        <tr><td>Predicted median housing value (1990 USD)</td>
            <td>$${prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td></tr>
      </table>`;
  } catch (e) {
    console.error('ONNX runtime error:', e);
    predictions.textContent = 'Error: ' + e.message;
  }
}

const genres = [
  'blues', 'classical', 'country', 'disco', 'hiphop',
  'jazz', 'metal', 'pop', 'reggae', 'rock'
];

const sampleSource = 'https://raw.githubusercontent.com/Harmanveer2546/Music-Genre-Classification/main/features_3_sec.csv';
const sampleList = [
  { filename: 'blues.00066.wav', audio: './audio/blues.00066.wav' },
  { filename: 'pop.00097.wav', audio: './audio/pop.00097.wav' },
  { filename: 'pop.00098.wav', audio: './audio/pop.00098.wav' },
  { filename: 'pop.00099.wav', audio: './audio/pop.00099.wav' },
  { filename: 'rock.00089.wav', audio: './audio/rock.00089.wav' },
  { filename: 'hiphop.00087.wav', audio: './audio/hiphop.00087.wav' },
  { filename: 'hiphop.00088.wav', audio: './audio/hiphop.00088.wav' },
  { filename: 'classical.00056.wav', audio: './audio/classical.00056.wav' },
  { filename: 'disco.00079.wav', audio: './audio/disco.00079.wav' },
  { filename: 'metal.00000.wav', audio: 'https://huggingface.co/datasets/m-a-p/GTZAN/resolve/main/genres/metal/metal.00000.wav' }
];

let musicSamples = [];

function setSampleStatus(message) {
  document.getElementById('sampleStatus').textContent = message;
}

function selectedSample() {
  return musicSamples[Number(document.getElementById('sampleSelector').value)];
}

function updateSelectedAudio() {
  const sample = selectedSample();
  const player = document.getElementById('samplePlayer');
  document.getElementById('classificationOutput').textContent = '';
  player.pause();
  if (!sample) {
    player.removeAttribute('src');
    player.load();
    return;
  }
  player.src = sample.audio;
  player.load();
  document.getElementById('classifyButton').disabled = !sample.values;
}

function parseGTZANSamples(csvText) {
  const rows = csvText.trim().split(/\r?\n/);
  const header = rows[0].split(',').map(s => s.trim().replace(/^"|"$/g, ''));
  const filenameColumn = header.indexOf('filename');
  const labelColumn = header.indexOf('label');
  const featureColumns = header.map((name, index) => ({ name, index }))
    .filter(c => c.name !== 'filename' && c.name !== 'label' && c.name !== 'length')
    .map(c => c.index);

  if (filenameColumn < 0 || labelColumn < 0 || featureColumns.length !== 57) {
    throw new Error('Expected GTZAN features_3_sec.csv with filename, label, and 57 numeric features.');
  }

  const byName = new Map(sampleList.map(item => [item.filename.slice(0, -4), null]));
  for (let i = 1; i < rows.length; i++) {
    const cells = rows[i].split(',');
    if (cells.length !== header.length) continue;
    const segment = cells[filenameColumn].trim().replace(/^"|"$/g, '');
    const base = segment.replace(/\.\d+\.wav$/i, '');
    if (!byName.has(base)) continue;
    if (byName.get(base) !== null) continue;
    const values = featureColumns.map(index => Number(cells[index]));
    if (values.some(value => !Number.isFinite(value))) continue;
    byName.set(base, { values, genre: cells[labelColumn].trim(), segment });
  }

  return sampleList.map(item => {
    const found = byName.get(item.filename.slice(0, -4));
    return { ...item, values: found ? found.values : null,
             genre: found ? found.genre : null,
             segment: found ? found.segment : null };
  });
}

function showMusicSamples(samples) {
  musicSamples = samples;
  const selector = document.getElementById('sampleSelector');
  selector.innerHTML = '';
  samples.forEach((sample, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `Sample ${index + 1}`;
    selector.appendChild(option);
  });
  selector.disabled = false;
  selector.onchange = updateSelectedAudio;
  updateSelectedAudio();
  const count = samples.filter(s => s.values).length;
  setSampleStatus(`${count} of ${samples.length} samples matched to GTZAN features. Select a sample, listen, then predict.`);
}

async function loadSampleCSV(input) {
  try {
    if (!input.files || !input.files[0]) return;
    showMusicSamples(parseGTZANSamples(await input.files[0].text()));
  } catch (e) {
    setSampleStatus('Could not read CSV: ' + e.message);
  }
}

async function loadGTZANSamples() {
  showMusicSamples(sampleList.map(item => ({...item, values: null, genre: null, segment: null})));
  try {
    const response = await fetch(sampleSource);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    showMusicSamples(parseGTZANSamples(await response.text()));
  } catch (e) {
    console.error('GTZAN dataset loading error:', e);
    setSampleStatus('Could not retrieve feature data. Select your local features_3_sec.csv below to enable predictions. Audio playback is still available.');
  }
}

async function runClassification() {
  const predictions = document.getElementById('classificationOutput');
  const sample = selectedSample();
  if (!sample || !sample.values) {
    predictions.textContent = 'Feature data unavailable. Load GTZAN features_3_sec.csv below.';
    return;
  }
  const x = new Float32Array(sample.values);
  const tensorX = new ort.Tensor('float32', x, [1, 57]);
  try {
    predictions.textContent = 'Running prediction...';
    const session = await ort.InferenceSession.create('./classification.onnx');
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;
    if (output.length !== genres.length) throw new Error('Unexpected output length');
    const index = Array.from(output).indexOf(Math.max(...output));
    predictions.innerHTML = `<table>
      <tr><td>Selected sample</td><td>Sample ${Number(document.getElementById('sampleSelector').value) + 1}</td></tr>
      <tr><td>Predicted music genre</td><td><strong>${genres[index]}</strong></td></tr>
      <tr><td>GTZAN reference genre</td><td>${sample.genre}</td></tr>
    </table>`;
  } catch (e) {
    console.error('ONNX runtime error:', e);
    predictions.textContent = 'Error: ' + e.message;
  }
}

document.addEventListener('DOMContentLoaded', loadGTZANSamples);

async function runExampleLEforward() {
    
  const x = new Float32Array(7);
  x[0] = parseFloat(document.getElementById('LEbox0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('LEbox1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('LEbox2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('LEbox3c1').value) || 0;
  x[4] = parseFloat(document.getElementById('LEbox4c1').value) || 0;
  x[5] = parseFloat(document.getElementById('LEbox5c1').value) || 0;
  x[6] = parseFloat(document.getElementById('LEbox6c1').value) || 0;  

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Forward.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 4

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsLEforward');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>tgt</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>hmt</td>           <td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>prod rate</td>          <td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>fta</td>          <td id="c1td3">${output[3].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    
}



async function runExampleLEinverse() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('LEbox0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('LEbox1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('LEbox2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('LEbox3c2').value) || 0;
   
  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Inverse_2.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsLEinverse');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>pci rate</td>           <td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>o2 vol fract</td>          <td id="c1td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>h2 temp</td>           <td id="c1td1">${output[4].toFixed(2)}</td></tr>
        <tr><td> hb temp </td>          <td id="c1td2">${output[5].toFixed(2)}</td></tr>
        <tr><td> wind rate</td>          <td id="c1td3">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    
}



async function runExample2() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_realData_new.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictions2');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c2td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c2td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c2td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c2td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c2td4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c2td5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c2td6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvg();
    
}






async function runExampleSynthetic() {
    
  const x = new Float32Array(8);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;
    
  x[4] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[5] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[6] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[7] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 8]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_syntheticData.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 8

    // render here (output is in scope)
    const predictions = document.getElementById('predSynthetic');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="syntd0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="syntd1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="syntd2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="syntd3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="syntd4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="syntd5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="syntd6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvg();
    
}





async function runExampleRealGenHong() {
    await runExampleHW1(); 
    await runExampleHW2();
    runAvgHW();  
}




async function runAvg() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2').innerHTML );
    var c1td3 = parseFloat( document.getElementById('c1td3').innerHTML );
    var c1td4 = parseFloat( document.getElementById('c1td4').innerHTML );
    var c1td5 = parseFloat( document.getElementById('c1td5').innerHTML );
    var c1td6 = parseFloat( document.getElementById('c1td6').innerHTML );
  
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2').innerHTML );
    var c2td3 = parseFloat( document.getElementById('c2td3').innerHTML );
    var c2td4 = parseFloat( document.getElementById('c2td4').innerHTML );
    var c2td5 = parseFloat( document.getElementById('c2td5').innerHTML );
    var c2td6 = parseFloat( document.getElementById('c2td6').innerHTML );

    
    td0 = (c1td0 + c2td0)/2;
    td1 = (c1td1 + c2td1)/2;
    td2 = (c1td2 + c2td2)/2;
    td3 = (c1td3 + c2td3)/2;
    td4 = (c1td4 + c2td4)/2;
    td5 = (c1td5 + c2td5)/2;
    td6 = (c1td6 + c2td6)/2;

 
     difference.innerHTML = `<hr> Average is: <br/> 
 <table>
 
  <tr>
  <td> i_h2i_rate </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_h2_temp </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_ngi_rate  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_pci_rate</td>
  <td> ${td3.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_o2_volfract </td>
  <td> ${td4.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_hbtemp  </td>
  <td> ${td5.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_wind_rt </td>
  <td> ${td6.toFixed(2)} </td>
  </tr> 
 
 </table>   `;
    
}


async function runExampleHW1() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;    // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsHW1');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c1td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c1td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c1td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c1td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c1td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c1td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c1td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    runAvgHW();
}



async function runExampleHW2() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsHW2');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c2td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c2td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c2td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c2td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c2td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c2td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c2td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvgHW();
    
}



async function runAvgHW() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0HW').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1HW').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2HW').innerHTML );
    var c1td3 = parseFloat( document.getElementById('c1td3HW').innerHTML );
    var c1td4 = parseFloat( document.getElementById('c1td4HW').innerHTML );
    var c1td5 = parseFloat( document.getElementById('c1td5HW').innerHTML );
    var c1td6 = parseFloat( document.getElementById('c1td6HW').innerHTML );
  
    
    var c2td0 = parseFloat( document.getElementById('c2td0HW').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1HW').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2HW').innerHTML );
    var c2td3 = parseFloat( document.getElementById('c2td3HW').innerHTML );
    var c2td4 = parseFloat( document.getElementById('c2td4HW').innerHTML );
    var c2td5 = parseFloat( document.getElementById('c2td5HW').innerHTML );
    var c2td6 = parseFloat( document.getElementById('c2td6HW').innerHTML );

    
    td0 = (c1td0 + c2td0)/2;
    td1 = (c1td1 + c2td1)/2;
    td2 = (c1td2 + c2td2)/2;
    td3 = (c1td3 + c2td3)/2;
    td4 = (c1td4 + c2td4)/2;
    td5 = (c1td5 + c2td5)/2;
    td6 = (c1td6 + c2td6)/2;

 
     differenceHW.innerHTML = `<hr> Average is: <br/> 
 <table>
 
  <tr>
  <td> i_h2i_rate </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_h2_temp </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_ngi_rate  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_pci_rate</td>
  <td> ${td3.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_o2_volfract </td>
  <td> ${td4.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_hbtemp  </td>
  <td> ${td5.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_wind_rt </td>
  <td> ${td6.toFixed(2)} </td>
  </tr> 
 
 </table>   `;
    
}

