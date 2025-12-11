const fs = require('fs');

// Configuration for the tuning process
const MAX_ITERATIONS = 20;
const SERVER_URL = 'http://localhost:8080/completion';

// Initial Model Parameters (starting point)
let modelConfig = {
  temperature: 0.7,
  top_k: 40,
  top_p: 0.9,
  min_p: 0.05,
  repeat_penalty: 1.1,
  presence_penalty: 0.0, // using snake_case for server payload
  frequency_penalty: 0.0,
  dry_multiplier: 0.0,
  max_tokens: 300 // limit generation length
};

// Questions to test
const questions = [
  "너 이름이 뭐지?",
  "넌 어디서 태어 났니?",
  "넌 누가 만들었어?",
  "대한민국의 수도에 대해서 알려줘."
];

// Analysis Criteria
const BANNED_PATTERNS = ['~~', '!!', 'ㅎㅎ', 'ㅋㅋ', 'LAPTOP', 'aaaa', '....', ';;;;'];
const MAX_SENTENCES = 5;

// Helper: Build Prompt (matches api.js logic)
function buildPrompt(userQuery) {
  const systemPrompt = [
    "당신은 유용한 AI 어시스턴트 '뤼(Luu)'입니다.",
    "사용자의 질문에 한국어로 답변하세요.",
    "답변은 반드시 3~5문장 이내로 명확하게 끝내세요."
  ].join(" ");

  return `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${userQuery}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}

// Helper: Call Llama Server
async function runGeneration(prompt, config) {
  const payload = {
    prompt,
    n_predict: config.max_tokens,
    temperature: config.temperature,
    top_k: config.top_k,
    top_p: config.top_p,
    min_p: config.min_p,
    repeat_penalty: config.repeat_penalty,
    presence_penalty: config.presence_penalty,
    frequency_penalty: config.frequency_penalty,
    dry_multiplier: config.dry_multiplier,
    stop: ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", ...BANNED_PATTERNS], // Server-side stop
    stream: false // We want the full response for analysis
  };

  try {
    const response = await fetch(SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    return data.content || "";
  } catch (error) {
    console.error("Error calling server:", error.message);
    return null;
  }
}

// Helper: Analyze Response
function analyzeResponse(text) {
  const issues = [];

  // 1. Repetition Check
  for (const pattern of BANNED_PATTERNS) {
    if (text.includes(pattern)) {
      issues.push(`Contains banned pattern: "${pattern}"`);
    }
  }

  // 1.1 Check for repeated substrings (primitive DRY check)
  // Check if any substring of length 20+ appears more than once
  for (let i = 0; i < text.length - 20; i++) {
      const substr = text.substr(i, 20);
      if (text.indexOf(substr, i + 20) !== -1) {
          issues.push(`Contains repeated phrase (DRY failure): "${substr}..."`);
          break; 
      }
  }

  // 2. Length Check (Sentences)
  // Simple heuristic: count sentence endings
  const sentences = text.match(/[.!?\n]+/g);
  const sentenceCount = sentences ? sentences.length : 0;
  if (sentenceCount > MAX_SENTENCES) {
    issues.push(`Too long: ${sentenceCount} sentences (max ${MAX_SENTENCES})`);
  }
  
  // 3. Length Check (Raw Chars) - Just in case it's huge
  if (text.length > 500) {
      issues.push(`Too long: ${text.length} characters`);
  }

  // 4. Hallucination/Identity Check
  if (text.includes("LAPTOP")) {
    issues.push("Identity hallucination: 'LAPTOP'");
  }

  return issues;
}

// Main Tuning Loop
async function runTuning() {
  console.log("Starting Auto-Tuning Process...");
  console.log("Target Server:", SERVER_URL);
  
  let iteration = 0;
  let optimalFound = false;

  while (iteration < MAX_ITERATIONS && !optimalFound) {
    iteration++;
    console.log(`\n--- Iteration ${iteration} ---`);
    console.log("Current Config:", JSON.stringify(modelConfig, null, 2));

    let allPassed = true;
    let accumulatedIssues = [];

    for (const q of questions) {
      process.stdout.write(`Testing: "${q}" ... `);
      const prompt = buildPrompt(q);
      const response = await runGeneration(prompt, modelConfig);
      
      if (response === null) {
        console.log("FAILED (Server Error)");
        return; // Stop if server is down
      }

      console.log(`\nResponse: "${response.trim()}"`);
      
      const issues = analyzeResponse(response);
      if (issues.length > 0) {
        console.log(`  [FAIL] Issues: ${issues.join(', ')}`);
        accumulatedIssues.push(...issues);
        allPassed = false;
        // Don't break immediately, let's see other questions to gather more data
      } else {
        console.log("  [PASS]");
      }
      // Small delay between requests
      await new Promise(r => setTimeout(r, 500));
    }

    if (allPassed) {
      optimalFound = true;
      console.log("\n✅ SUCCESS! Optimal configuration found.");
    } else {
      console.log("\n❌ Iteration Failed. Adjusting parameters...");
      adjustParameters(accumulatedIssues);
    }
  }

  if (optimalFound) {
    console.log("\n=== Final Optimized Configuration ===");
    console.log(JSON.stringify(modelConfig, null, 2));
    
    // Write to a file for easy copying
    fs.writeFileSync('optimized_config.json', JSON.stringify(modelConfig, null, 2));
    console.log("Saved to optimized_config.json");
  } else {
    console.log("\n⚠️ Max iterations reached without finding perfect config.");
    console.log("Last Config:", JSON.stringify(modelConfig, null, 2));
  }
}

// Helper: Adjust Parameters based on issues
function adjustParameters(issues) {
  // Simple heuristic adjustments
  const hasRepetition = issues.some(i => i.includes("banned pattern"));
  const hasLengthIssue = issues.some(i => i.includes("Too long"));
  const hasIdentityIssue = issues.some(i => i.includes("Identity"));

  if (hasRepetition) {
    console.log("  -> Increasing Repeat Penalty & Frequency Penalty");
    modelConfig.repeat_penalty = parseFloat((modelConfig.repeat_penalty + 0.05).toFixed(2));
    // Cap at reasonable max
    if (modelConfig.repeat_penalty > 1.3) modelConfig.repeat_penalty = 1.3;
    
    // Also try DRY if simple penalty isn't working
    if (modelConfig.repeat_penalty > 1.15 && modelConfig.dry_multiplier === 0.0) {
        console.log("  -> Enabling DRY Sampling");
        modelConfig.dry_multiplier = 0.5;
    }
  }

  if (hasLengthIssue) {
    console.log("  -> Increasing Min P (to cut off low-prob tokens) & Reducing Max Tokens");
    modelConfig.min_p = parseFloat((modelConfig.min_p + 0.02).toFixed(2));
    if (modelConfig.min_p > 0.2) modelConfig.min_p = 0.2;
    
    // Reduce max_tokens slightly to force brevity via hard limit if needed
    // But mainly rely on min_p and stop tokens
  }

  if (hasIdentityIssue) {
    console.log("  -> Decreasing Temperature (more deterministic)");
    modelConfig.temperature = parseFloat((modelConfig.temperature - 0.05).toFixed(2));
    if (modelConfig.temperature < 0.1) modelConfig.temperature = 0.1;
  }
  
  // General adjustment if nothing specific but still failing (e.g. slight incoherence)
  // or just to perturb the model
  if (!hasRepetition && !hasLengthIssue && !hasIdentityIssue) {
       console.log("  -> General Tweak: Slightly increasing Top P");
       modelConfig.top_p = parseFloat((modelConfig.top_p - 0.05).toFixed(2)); // tighten top_p
  }
}

// Run the script
runTuning();

