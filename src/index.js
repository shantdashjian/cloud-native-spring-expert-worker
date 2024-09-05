import { ChatOpenAI } from "langchain/chat_models/openai"
import { PromptTemplate } from "langchain/prompts"
import { StringOutputParser } from 'langchain/schema/output_parser'
import { combineDocuments } from "./utils/combineDocuments"
import { RunnablePassthrough, RunnableSequence } from "langchain/schema/runnable"
import formatConversationHistory from "./utils/formatConversationHistory"
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { createClient } from '@supabase/supabase-js'

const corsHeaders = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'POST, OPTIONS',
	'Access-Control-Allow-Headers': 'Content-Type',
}

export default {
	async fetch(request, env, ctx) {
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				status: 200,
				headers: corsHeaders
			})
		}

		const openAIApiKey = env.OPENAI_API_KEY

		const embeddings = new OpenAIEmbeddings({ openAIApiKey })
		const supabaseApiKey = env.SUPABASE_API_KEY
		const supabaseProjectUrl = env.SUPABASE_PROJECT_URL
		const client = createClient(supabaseProjectUrl, supabaseApiKey)

		const vectorStore = new SupabaseVectorStore(embeddings, {
			client,
			tableName: 'cloud_native_spring_documents',
			queryName: 'match_cloud_native_spring_documents'
		})

		const retriever = vectorStore.asRetriever(6)

		const llm = new ChatOpenAI({ openAIApiKey })

		const standaloneQuestionTemplate = `Given a question and the converstaion history, 
							convert the question to a standalone question. You can use the converstaion history as 
							a resource as well. 
							question: {question} 
							conversation history: {conversation_history}
							standalone question: `

		const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

		const answerTemplate = `You are a friendly and enthusiastic expert who can answer a given 
							question about Cloud Native Spring based on the context provided. 
							Try to find the answer in the context as it is the primary source of knowledge. 
							You could also use the conversation history if you cannot find the answer in the context. 
							If you really don't know the answer, say "I'm sorry, 
							I don't know the answer to that." Don't try to make up an answer. 
							Be friendly and make the response conversational and relatively short.
							context: {context}
							conversation history: {conversation_history}
							question: {question}
							answer: `

		const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

		const standaloneQuestionChain = standaloneQuestionPrompt
			.pipe(llm)
			.pipe(new StringOutputParser())

		const retrieverChain = RunnableSequence.from([
			prevResult => prevResult.standalone_question,
			retriever,
			combineDocuments
		])

		const answerChain = answerPrompt
			.pipe(llm)
			.pipe(new StringOutputParser())

		const chain = RunnableSequence.from([
			{
				standalone_question: standaloneQuestionChain,
				original_input: new RunnablePassthrough()
			},
			{
				context: retrieverChain,
				question: ({ original_input }) => original_input.question,
				conversation_history: ({ original_input }) => original_input.conversation_history
			},
			answerChain
		])

		try {
			const requestBody = await request.json()
			const question = requestBody.question
			const conversationHistory = requestBody.conversationHistory

			const response = await chain.invoke({
				question: question,
				conversation_history: formatConversationHistory(conversationHistory)
			})
			return new Response(JSON.stringify(response), { headers: corsHeaders })
		} catch (error) {
			return new Response(JSON.stringify({ error: error }), { status: 500, headers: corsHeaders })
		}
	},
};
