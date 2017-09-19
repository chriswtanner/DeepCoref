class StanLink:
	def __init__(self, parent, child, relationship):
		self.parent = parent
		self.child = child
		self.relationship = relationship

	def __str__(self):
		return(str(self.parent) + " --(" + str(self.relationship) + ")--> " + str(self.child))
