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

// Use existing vector store
const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "pdf-rag-new",
});

// Then add your documents
await vectorStore.addDocuments(splitPDF);

// Define a prompt template
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that answers questions based on the provided context.\nIf the information cannot be found in the context, say you don't know.\n\nContext: {context}"],
  ["human", "{question}"]
]);

// Multi Query Prompt
const multiQueryPrompt = ChatPromptTemplate.fromMessages([
  ["system", `You are a helpful assistant that generates different versions of a search query to help retrieve relevant information.
  
Generate 3 different versions of the given search query. Each version should:
- Rephrase the original query
- Focus on different aspects of the question
- Use different terminology/synonyms
- Try alternative phrasings

Return ONLY a JSON array of strings with no explanation. Example: ["query 1", "query 2", "query 3"]`],
  ["human", "{question}"]
]);

async function multiQueryRetrieval(question) {
  // Generate multiple query variations
  const queryGenMessages = await multiQueryPrompt.invoke({
    question: question
  });

  const queryGenResponse = await model.invoke(queryGenMessages);
  console.log('Generated Response:', queryGenResponse.content);
  
  // Parse and display the variations
  try {
    // Clean the response of any markdown code blocks
    let cleanedResponse = queryGenResponse.content;
    
    // Remove markdown code block syntax if present
    if (cleanedResponse.includes("```")) {
      // Extract the content between code block markers
      cleanedResponse = cleanedResponse.replace(/```(json|javascript)?\s*/g, '').replace(/```\s*$/g, '');
    }
    
    // Trim any whitespace
    cleanedResponse = cleanedResponse.trim();
    
    console.log('Cleaned Response:', cleanedResponse);
    
    const queryVariations = JSON.parse(cleanedResponse);
    console.log('Parsed Query Variations:', queryVariations);
    
    // Just for testing - show all variations
    console.log('Original Question:', question);
    console.log('Generated Variations:');
    queryVariations.forEach((query, index) => {
      console.log(`${index + 1}. ${query}`);
    });
    
    // For now, just return the original question's retrieval results
    const retrievedDocs = await vectorStore.similaritySearch(question);
    return retrievedDocs;
  } catch (e) {
    console.error('Error parsing query variations:', e);
    console.error('Attempted to parse:', queryGenResponse.content);
    // Fallback to original query
    const retrievedDocs = await vectorStore.similaritySearch(question);
    return retrievedDocs;
  }
}

async function main() {
  let question = "What is Nodejs";
  
  // Use multi-query retrieval
  const retrievedDocs = await multiQueryRetrieval(question);
  
  const docsContent = retrievedDocs.map((doc) => doc.pageContent).join("\n");
  
  const messages = await promptTemplate.invoke({
    question: question,
    context: docsContent,
  });
  
  const answer = await model.invoke(messages);
  console.log('Final Answer:', answer.content);
}

main();