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

async function multiQueryRetrieval(question, maxDocsPerQuery = 3) {
  // Generate multiple query variations
  const queryGenMessages = await multiQueryPrompt.invoke({
    question: question
  });

  const queryGenResponse = await model.invoke(queryGenMessages);
  console.log('Generated Response:', queryGenResponse.content);
  
  try {
    // Clean the response of any markdown code blocks
    let cleanedResponse = queryGenResponse.content;
    
    if (cleanedResponse.includes("```")) {
      cleanedResponse = cleanedResponse.replace(/```(json|javascript)?\s*/g, '').replace(/```\s*$/g, '');
    }
    
    cleanedResponse = cleanedResponse.trim();
    
    const queryVariations = JSON.parse(cleanedResponse);
    console.log('Parsed Query Variations:', queryVariations);
    
    // Add the original query to the variations
    const allQueries = [question, ...queryVariations];
    
    console.log('All Queries to Use:');
    allQueries.forEach((query, index) => {
      console.log(`${index}: ${query}`);
    });
    
    // Perform retrievals for each query variation
    console.log('Retrieving documents for each query variation...');
    
    const allRetrievedDocs = [];
    
    for (const query of allQueries) {
      console.log(`Retrieving for: "${query}"`);
      const docs = await vectorStore.similaritySearch(query, maxDocsPerQuery);
      console.log(`Retrieved ${docs.length} documents`);
      
      // Add retrieved docs to our collection
      allRetrievedDocs.push(...docs);
    }
    
    console.log(`Total documents retrieved across all queries: ${allRetrievedDocs.length}`);
    
    return allRetrievedDocs;
    
  } catch (e) {
    console.error('Error in multi-query retrieval:', e);
    console.error('Attempted to parse:', queryGenResponse.content);
    
    // Fallback to original query
    console.log('Falling back to original query retrieval');
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