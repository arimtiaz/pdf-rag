//Multi Query
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


// Define a prompt template
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that answers questions based on the provided context.\nIf the information cannot be found in the context, say you don't know.\n\nContext: {context}"],
  ["human", "{question}"]
]);

async function main() {
  let question = "Which application was built at the end of this course";

  const retrievedDocs = await vectorStore.similaritySearch(question);
  const docsContent = retrievedDocs.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: question,
    context: docsContent,
  });
  const answer = await model.invoke(messages);
  console.log(answer.content)
}
main()