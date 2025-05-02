// Decomposition
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "dotenv/config";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "gemini-2.0-flash",
});

//   Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  apiKey: process.env.GOOGLE_API_KEY,
});

// PDF Loader
const sourcePdf = "./PDF-Guide-Node-Andrew-Mead-v3.pdf";
const loader = new PDFLoader(sourcePdf);
const docs = await loader.load();

// PDF Splitting
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500, // Increase from 500
  chunkOverlap: 200, // Increase overlap slightly
});

const splitPDF = await textSplitter.splitDocuments(docs);

// Index chunks
// const vectorStore = await QdrantVectorStore.fromDocuments(
//   splitPDF,
//   embeddings,
//   {
//     url: process.env.QDRANT_URL,
//     collectionName: 'pdf-rag-new', // Use a new name
//   }
// );

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "pdf-rag-new",
});

// Then add your documents
await vectorStore.addDocuments(splitPDF);



const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that answers questions based on the provided context.\nIf the information cannot be found in the context, say you don't know.\n\nContext: {context}"],
  ["human", "{question}"]
]);

// NEW FUNCTION: Query decomposition
async function decomposeQuery(complexQuery) {
  const decompositionPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an AI assistant that helps break down complex questions into simpler sub-questions. Return only an array of 2-4 sub-questions in JSON format."],
    ["human", `Break down this complex query into simple sub-queries that together help answer the original question: "${complexQuery}". 
    Format your response as a valid JSON array of strings. For example: ["sub-question 1", "sub-question 2", "sub-question 3"]`]
  ]);

  const messages = await decompositionPrompt.invoke({});
  const response = await model.invoke(messages);
  
  try {
    // Parse the response to extract the JSON array
    const responseText = response.content.toString();
    // Find anything that looks like a JSON array in the response
    const jsonMatch = responseText.match(/\[.*\]/s);
    
    if (jsonMatch) {
      const subQueries = JSON.parse(jsonMatch[0]);
      console.log("Decomposed into:", subQueries);
      return subQueries;
    } else {
      // Fallback if no JSON array is found
      console.log("Could not parse sub-queries, using original query");
      return [complexQuery];
    }
  } catch (error) {
    console.error("Error parsing decomposed queries:", error);
    return [complexQuery]; // Fallback to original query
  }
}

// NEW FUNCTION: Get unique documents from multiple searches
function deduplicateDocuments(documents) {
  const uniqueDocs = [];
  const seenContents = new Set();
  
  for (const doc of documents) {
    // Use a simple approach to identify duplicate documents
    if (!seenContents.has(doc.pageContent)) {
      seenContents.add(doc.pageContent);
      uniqueDocs.push(doc);
    }
  }
  
  return uniqueDocs;
}

// MODIFIED: Main function using decomposition
async function main() {
  let complexQuestion = "What are the key features of Node.js and how do you build a weather application with it?";
  
  // 1. Decompose the complex query
  const subQueries = await decomposeQuery(complexQuestion);
  
  // 2. Retrieve documents for each sub-query
  let allRetrievedDocs = [];
  
  for (const query of subQueries) {
    console.log(`Searching for: "${query}"`);
    const docs = await vectorStore.similaritySearch(query, 3); // Get top 3 docs for each sub-query
    allRetrievedDocs = allRetrievedDocs.concat(docs);
  }
  
  // 3. Deduplicate documents to avoid repetition
  const uniqueDocs = deduplicateDocuments(allRetrievedDocs);
  console.log(`Retrieved ${allRetrievedDocs.length} total docs, ${uniqueDocs.length} after deduplication`);
  
  // 4. Generate the final answer using all relevant documents
  const docsContent = uniqueDocs.map((doc) => doc.pageContent).join("\n");
  
  const messages = await promptTemplate.invoke({
    question: complexQuestion, // Use the original question
    context: docsContent,      // But provide context from all sub-queries
  });
  
  const answer = await model.invoke(messages);
  console.log("\nFINAL ANSWER:");
  console.log(answer.content);
}

main();