		for dir_num in hddcrp_parsed.dirToDocs:
			print("dir:",dir_num)
			refToHMs = defaultdict(list)
			for doc_id in hddcrp_parsed.dirToDocs[dir_num]:
				for hm in hddcrp_parsed.docToHMentions[doc_id]:
					ref_id = hm.ref_id
					refToHMs[ref_id].append(hm)

			for ref_id in refToHMs:
				print("ref:",ref_id)
				for hm in refToHMs[ref_id]:
					print(hm)
			exit(1)


		for ref_id in corpus.refToDMs:
			print("ref:",str(ref_id))
			for dm in corpus.refToDMs[ref_id]:
				print("dm:",dm,str(corpus.dmToMention[dm]))

		exit(1)
		for dir_num in corpus.dirToDocs:
			print(dir_num,":")
			for doc_id in corpus.dirToDocs[dir_num]:
				print("doc:",str(doc_id))
				#for dm in corpus.docToDMs[doc_id]:
		exit(1)