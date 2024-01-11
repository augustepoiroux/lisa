package lisa.prooflib

import lisa.prooflib.BasicStepTactic.Rewrite
import lisa.prooflib.BasicStepTactic.*
import lisa.prooflib.ProofTacticLib.*
import lisa.prooflib.SimpleDeducedSteps.*
import lisa.prooflib.*
import lisa.utils.KernelHelpers.{_, given}
import lisa.utils.LisaException
import lisa.utils.UserLisaException
import lisa.utils.parsing.FOLPrinter
import lisa.utils.{_, given}

import scala.annotation.targetName

trait ProofsHelpers {
  library: Library & WithTheorems =>

  import lisa.fol.FOL.{given, *}

  given Library = library

  def extractProofStates(_proof: library.Proof, line: sourcecode.Line, file: sourcecode.File) = {
    // // let's print line and file
    // println(s"Line: ${line.value}\tFile: ${file.value}")
    // // and let's print the proof
    // println(s"Proof of " + _proof.owningTheorem.prettyGoal + ": ")
    // val prettyProofstep = ProofPrinter.prettySimpleProof(_proof, 0).split("\n").last
    // println(prettyProofstep)

    // /// Export the proof annotations to a file copy
    // // check that it ends with .scala
    // if (!file.value.endsWith(".scala")) {
    //   throw new Exception("The file name must end with .scala")
    // }
    // val annotatedFileName = file.value.dropRight(6) + "_annotated.sc"
    // val annotatedFile = new java.io.File(annotatedFileName)

    // // Get source file content
    // val source = scala.io.Source.fromFile(file.value)
    // val sourceLines = source.getLines.toArray
    // source.close()

    // val annotatedLine = sourceLines(line.value - 1) + " // " + prettyProofstep
    // println(annotatedLine)
    // println("-" * 120)

    // // open a copy of file, create one if it doesn't exist
    // if (!annotatedFile.exists()) {
    //   val fileWriter = new java.io.FileWriter(annotatedFile)
    //   fileWriter.write(sourceLines.mkString("\n"))
    //   fileWriter.close()
    // }

    // // open the file again to write the annotation at the right line
    // val prevAnnotation = scala.io.Source.fromFile(annotatedFile)
    // val lines = prevAnnotation.getLines.toArray
    // prevAnnotation.close()
    // val fileWriter = new java.io.FileWriter(annotatedFile)
    // lines(line.value - 1) = annotatedLine
    // fileWriter.write(lines.mkString("\n"))
    // fileWriter.close()
  }

  class HaveSequent private[ProofsHelpers] (bot: Sequent) {
    val x: lisa.fol.FOL.Sequent = bot
    inline infix def by(using proof: library.Proof, line: sourcecode.Line, file: sourcecode.File): By { val _proof: proof.type } = By(proof, line, file).asInstanceOf

    class By(val _proof: library.Proof, line: sourcecode.Line, file: sourcecode.File) {

      private val bot = HaveSequent.this.bot ++ (F.iterable_to_set(_proof.getAssumptions) |- ())
      inline infix def apply(tactic: Sequent => _proof.ProofTacticJudgement): _proof.ProofStep & _proof.Fact = {
        val res = tactic(bot).validate(line, file)
        extractProofStates(_proof, line, file)
        res
      }
      inline infix def apply(tactic: ProofSequentTactic): _proof.ProofStep = {
        val res = tactic(using library, _proof)(bot).validate(line, file)
        extractProofStates(_proof, line, file)
        res
      }
    }

    infix def subproof(using proof: Library#Proof, line: sourcecode.Line, file: sourcecode.File)(computeProof: proof.InnerProof ?=> Unit): proof.ProofStep = {
      val botWithAssumptions = HaveSequent.this.bot ++ (proof.getAssumptions |- ())
      val iProof: proof.InnerProof = new proof.InnerProof(Some(botWithAssumptions))
      computeProof(using iProof)
      (new BasicStepTactic.SUBPROOF(using proof)(Some(botWithAssumptions))(iProof)).judgement.validate(line, file).asInstanceOf[proof.ProofStep]
    }

  }

  class AndThenSequent private[ProofsHelpers] (bot: Sequent) {

    inline infix def by(using proof: library.Proof, line: sourcecode.Line, file: sourcecode.File): By { val _proof: proof.type } =
      By(proof, line, file).asInstanceOf[By { val _proof: proof.type }]

    class By(val _proof: library.Proof, line: sourcecode.Line, file: sourcecode.File) {
      private val bot = AndThenSequent.this.bot ++ (_proof.getAssumptions |- ())
      inline infix def apply(tactic: _proof.Fact => Sequent => _proof.ProofTacticJudgement): _proof.ProofStep = {
        val res = tactic(_proof.mostRecentStep)(bot).validate(line, file)
        extractProofStates(_proof, line, file)
        res
      }

      inline infix def apply(tactic: ProofFactSequentTactic): _proof.ProofStep = {
        val res = tactic(using library, _proof)(_proof.mostRecentStep)(bot).validate(line, file)
        extractProofStates(_proof, line, file)
        res
      }
    }
  }

  /**
   * Claim the given Sequent as a ProofTactic, which may require a justification by a proof tactic and premises.
   */
  def have(using proof: library.Proof)(res: Sequent): HaveSequent = HaveSequent(res)

  def have(using line: sourcecode.Line, file: sourcecode.File)(using proof: library.Proof)(v: proof.Fact | proof.ProofTacticJudgement) = v match {
    case judg: proof.ProofTacticJudgement => judg.validate(line, file)
    case fact: proof.Fact @unchecked => HaveSequent(proof.sequentOfFact(fact)).by(using proof, line, file)(Rewrite(using library, proof)(fact))
  }

  /**
   * Claim the given Sequent as a ProofTactic directly following the previously proven tactic,
   * which may require a justification by a proof tactic.
   */
  def thenHave(using proof: library.Proof)(res: Sequent): AndThenSequent = AndThenSequent(res)

  infix def andThen(using proof: library.Proof, line: sourcecode.Line, file: sourcecode.File): AndThen { val _proof: proof.type } = AndThen(proof, line, file).asInstanceOf

  class AndThen private[ProofsHelpers] (val _proof: library.Proof, line: sourcecode.Line, file: sourcecode.File) {
    inline infix def apply(tactic: _proof.Fact => _proof.ProofTacticJudgement): _proof.ProofStep = {
      tactic(_proof.mostRecentStep).validate(line, file)
    }
    inline infix def apply(tactic: ProofFactTactic): _proof.ProofStep = {
      tactic(using library, _proof)(_proof.mostRecentStep).validate(line, file)
    }
  }

  /*
  /**
   * Assume the given formula in all future left hand-side of claimed sequents.
   */
  def assume(using proof: library.Proof)(f: Formula): proof.ProofStep = {
    proof.addAssumption(f)
    have(() |- f) by BasicStepTactic.Hypothesis
  }
   */
  /**
   * Assume the given formulas in all future left hand-side of claimed sequents.
   */
  def assume(using proof: library.Proof)(fs: Formula*): proof.ProofStep = {
    fs.foreach(f => proof.addAssumption(f))
    have(() |- fs.toSet) by BasicStepTactic.Hypothesis
  }

  def thesis(using proof: library.Proof): Sequent = proof.possibleGoal.get
  def goal(using proof: library.Proof): Sequent = proof.possibleGoal.get

  def lastStep(using proof: library.Proof): proof.ProofStep = proof.mostRecentStep

  def sorry(using proof: library.Proof): proof.ProofStep = have(thesis) by Sorry

  def showCurrentProof(using om: OutputManager, _proof: library.Proof)(): Unit = {
    om.output("Current proof of " + _proof.owningTheorem.prettyGoal + ": ")
    om.output(
      lisa.utils.parsing.ProofPrinter.prettyProof(_proof, 2)
    )
  }

  extension (using proof: library.Proof)(fact: proof.Fact) {
    def of(insts: (F.SubstPair | F.Term)*): proof.InstantiatedFact = {
      proof.InstantiatedFact(fact, insts)
    }
    def statement: F.Sequent = proof.sequentOfFact(fact)
  }

  def currentProof(using p: library.Proof): Library#Proof = p

  ////////////////////////////////////////
  //  DSL for definitions and theorems  //
  ////////////////////////////////////////

  class UserInvalidDefinitionException(val symbol: String, errorMessage: String)(using line: sourcecode.Line, file: sourcecode.File) extends UserLisaException(errorMessage) { // TODO refine
    val showError: String = {
      val source = scala.io.Source.fromFile(file.value)
      val textline = source.getLines().drop(line.value - 1).next().dropWhile(c => c.isWhitespace)
      source.close()
      s"   Definition of $symbol at.(${file.value.split("/").last.split("\\\\").last}:${line.value}) is invalid:\n" +
        "   " + Console.RED + textline + Console.RESET + "\n\n" +
        "   " + errorMessage
    }
  }

  class The(val out: Variable, val f: Formula)(
      val just: JUSTIFICATION
  )
  class definitionWithVars[N <: Arity](val args: Seq[Variable]) {

    // inline infix def -->(using om: OutputManager, name: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(t: Term) = simpleDefinition(lambda(args, t, args.length))
    // inline infix def -->(using om: OutputManager, name: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(f: Formula) = predicateDefinition(lambda(args, f, args.length))

    inline infix def -->(using om: OutputManager, name: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(t: The): ConstantTermLabelOfArity[N] =
      val definition = FunctionDefinition[N](name.value, line.value, file.value)(args, t.out, t.f, t.just)
      exportJson(definition, "FunctionDefinition")
      definition.label

    inline infix def -->(using om: OutputManager, name: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(term: Term): ConstantTermLabelOfArity[N] =
      val definition = SimpleFunctionDefinition[N](name.value, line.value, file.value)(lambda(args, term).asInstanceOf)
      exportJson(definition, "FunctionDefinition")
      definition.label

    inline infix def -->(using om: OutputManager, name: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(formula: Formula): ConstantAtomicLabelOfArity[N] =
      val definition = PredicateDefinition[N](name.value, line.value, file.value)(lambda(args, formula).asInstanceOf)
      exportJson(definition, "PredicateDefinition")
      definition.label

    /**
     * Export the definition information in a json file
     */
    def exportJson(using om: OutputManager, fullName: sourcecode.FullName, line: sourcecode.Line, file: sourcecode.File)(definition: DEFINITION, kind: String): Unit = {
      val shortName = fullName.value.split("\\.").last
      val fileSubdir = fullName.value.split("\\.").dropRight(1).mkString("/")
      val jsonFilename = "data_extract/" + fileSubdir + ".json"
      val jsonFile = new java.io.File(jsonFilename)

      /// Check folder / file exists
      if (!jsonFile.getParentFile.exists()) {
        jsonFile.getParentFile.mkdirs()
      }
      if (!jsonFile.exists()) {
        jsonFile.createNewFile()
        // add {} to the file
        val pw = new java.io.PrintWriter(jsonFile)
        pw.write("{}")
        pw.close()
      }

      /// Extract the definition documentation if it exists from the source file (handcrafted extraction)
      var docstring = ""
      val sourceFile = scala.io.Source.fromFile(file.value)
      val lines =
        try sourceFile.getLines.toList
        finally sourceFile.close()

      // Get first line of the declaration
      var firstLine = line.value - 1
      while (firstLine >= 0 && !lines(firstLine).contains("def ") && !lines(firstLine).contains("val "))
        firstLine -= 1
        if (lines(firstLine).contains("*/"))
          throw new Exception(s"Error during extraction of the documentation of the definition $fullName (line ${line.value}) in file $fileSubdir)")

      // look for the first previous line containing "*/" and check that there are only whitespaces between this line and the theorem declaration
      var lastLine = firstLine - 1
      while (lastLine >= 0 && !lines(lastLine).contains("*/") && lines(lastLine).trim().isEmpty)
        lastLine -= 1
      if (lastLine >= 0 && lines(lastLine).contains("*/")) {
        // look for the first line containing "/**" before the theorem name and extract everything between "/**" and "*/"
        var firstLineDocstring = lastLine
        while (firstLineDocstring >= 0 && !lines(firstLineDocstring).contains("/**"))
          firstLineDocstring -= 1
        if (firstLineDocstring >= 0 && lines(firstLineDocstring).contains("/**"))
          docstring = lines.slice(firstLineDocstring, lastLine + 1).mkString("\n").stripIndent()
      }

      /// Extract definition statement from the source file (handcrafted extraction)
      // in order to achieve that, we start from the line of definition declaration and look until the first "/**", "val" or "def"
      var declaration = lines(firstLine)
      var i = firstLine + 1
      while (i < lines.length && !lines(i).contains("/**") && !lines(i).contains("val ") && !lines(i).contains("def ")) {
        if (!lines(i).trim().isEmpty)
          declaration += "\n" + lines(i)
        i += 1
      }
      declaration = declaration.stripIndent()

      /// Extract definitions used in the definition statement
      def extractFormulaDefs(formula: F.Formula): List[String] = {
        (formula.underlying.constantTermLabels.map(l => l.id.name).toList ++
          formula.underlying.constantPredicateLabels.map(l => l.id.name).toList).filter(_ != shortName)
      }
      val defs = (definition.statement.left.flatMap(f => extractFormulaDefs(f)) ++ definition.statement.right.flatMap(f => extractFormulaDefs(f))).toList

      /// Update json file content
      val jsonContent = ujson.read(jsonFile)
      val defId = shortName
      jsonContent(defId) = ujson.Obj(
        "line" -> line.value,
        "file_path" -> fileSubdir,
        "kind" -> kind,
        "docstring" -> docstring,
        "statement" -> lisa.utils.FOLPrinter.prettySequent(definition.statement.underlying),
        "declaration" -> declaration,
        "definitions" -> defs
      )
      val jsonWriter = new java.io.PrintWriter(jsonFile)
      ujson.writeTo(jsonContent, jsonWriter, indent = 4)
      jsonWriter.close()
    }
  }

  def DEF(): definitionWithVars[0] = new definitionWithVars[0](Nil)
  def DEF(a: Variable): definitionWithVars[1] = new definitionWithVars[1](Seq(a))
  def DEF(a: Variable, b: Variable): definitionWithVars[2] = new definitionWithVars[2](Seq(a, b))
  def DEF(a: Variable, b: Variable, c: Variable): definitionWithVars[3] = new definitionWithVars[3](Seq(a, b, c))
  def DEF(a: Variable, b: Variable, c: Variable, d: Variable): definitionWithVars[4] = new definitionWithVars[4](Seq(a, b, c, d))
  def DEF(a: Variable, b: Variable, c: Variable, d: Variable, e: Variable): definitionWithVars[5] = new definitionWithVars[5](Seq(a, b, c, d, e))
  def DEF(a: Variable, b: Variable, c: Variable, d: Variable, e: Variable, f: Variable): definitionWithVars[6] = new definitionWithVars[6](Seq(a, b, c, d, e, f))
  def DEF(a: Variable, b: Variable, c: Variable, d: Variable, e: Variable, f: Variable, g: Variable): definitionWithVars[7] = new definitionWithVars[7](Seq(a, b, c, d, e, f, g))

  // def DEF: definitionWithVars[0] = new definitionWithVars[0](EmptyTuple) //todo conversion to empty tuple gets bad type

  // Definition helpers, not part of the DSL

  /**
   * Allows to make definitions "by unique existance" of a function symbol
   */
  class FunctionDefinition[N <: F.Arity](using om: OutputManager)(val fullName: String, line: Int, file: String)(
      val vars: Seq[F.Variable],
      val out: F.Variable,
      val f: F.Formula,
      j: JUSTIFICATION
  ) extends DEFINITION(line, file) {
    def funWithArgs = label.applySeq(vars)
    override def repr: String =
      s" ${if (withSorry) " Sorry" else ""} Definition of function symbol ${funWithArgs} := the ${out} such that ${(out === funWithArgs) <=> f})\n"

    // val expr = LambdaExpression[Term, Formula, N](vars, f, valueOf[N])

    lazy val label: ConstantTermLabelOfArity[N] = (if (vars.length == 0) F.Constant(name) else F.ConstantFunctionLabel[N](name, vars.length.asInstanceOf)).asInstanceOf

    val innerJustification: theory.FunctionDefinition = {
      val conclusion: F.Sequent = j.statement
      val pr: SCProof = SCProof(IndexedSeq(SC.Restate(conclusion.underlying, -1)), IndexedSeq(conclusion.underlying))
      if (!(conclusion.left.isEmpty && (conclusion.right.size == 1))) {
        om.lisaThrow(
          UserInvalidDefinitionException(
            name,
            s"The given justification is not valid for a definition" +
              s"The justification should be of the form ${(() |- F.BinderFormula(F.ExistsOne, out, F.VariableFormula("phi")))}" +
              s"instead of the given ${conclusion.underlying}"
          )
        )
      }
      if (!(f.freeSchematicLabels.subsetOf(vars.toSet + out))) {
        om.lisaThrow(
          UserInvalidDefinitionException(
            name,
            s"The definition is not allowed to contain schematic symbols or free variables." +
              s"The symbols {${(f.freeSchematicLabels -- vars.toSet - out).mkString(", ")}} are free in the expression ${f.toString}."
          )
        )
      }
      val proven = conclusion.right.head match {
        case F.BinderFormula(F.ExistsOne, bound, inner) => inner
        case F.BinderFormula(F.Exists, x, F.BinderFormula(F.Forall, y, F.AppliedConnector(F.Iff, Seq(l, r)))) if F.isSame(l, x === y) => r
        case F.BinderFormula(F.Exists, x, F.BinderFormula(F.Forall, y, F.AppliedConnector(F.Iff, Seq(l, r)))) if F.isSame(r, x === y) => l
        case _ =>
          om.lisaThrow(
            UserInvalidDefinitionException(
              name,
              s"The given justification is not valid for a definition" +
                s"The justification should be of the form ${(() |- F.BinderFormula(F.ExistsOne, out, F.VariableFormula("phi")))}" +
                s"instead of the given ${conclusion.underlying}"
            )
          )
      }
      val underf = f.underlying
      val undervars = vars.map(_.underlyingLabel)
      val ulabel = K.ConstantFunctionLabel(name, vars.size)
      val judgement = theory.makeFunctionDefinition(pr, Seq(j.innerJustification), ulabel, out.underlyingLabel, K.LambdaTermFormula(undervars, underf), proven.underlying)
      judgement match {
        case K.ValidJustification(just) =>
          just
        case wrongJudgement: K.InvalidJustification[?] =>
          if (!theory.belongsToTheory(underf)) {
            import K.findUndefinedSymbols
            om.lisaThrow(
              UserInvalidDefinitionException(
                name,
                s"All symbols in the definition must belong to the theory. The symbols ${theory.findUndefinedSymbols(underf)} are unknown and you need to define them first."
              )
            )
          }
          if (!theory.isAvailable(ulabel)) {
            om.lisaThrow(UserInvalidDefinitionException(name, s"The symbol ${name} has already been defined and can't be redefined."))
          }
          if (!(underf.freeSchematicTermLabels.subsetOf(undervars.toSet + out.underlyingLabel) && underf.schematicFormulaLabels.isEmpty)) {
            om.lisaThrow(
              UserInvalidDefinitionException(
                name,
                s"The definition is not allowed to contain schematic symbols or free variables." +
                  s"Kernel returned error: The symbols {${(underf.freeSchematicTermLabels -- undervars.toSet - out.underlyingLabel ++ underf.freeSchematicTermLabels)
                      .mkString(", ")}} are free in the expression ${underf.toString}."
              )
            )
          }
          om.lisaThrow(
            LisaException.InvalidKernelJustificationComputation(
              "The final proof was rejected by LISA's logical kernel. This may be due to a faulty proof computation or an error in LISA.",
              wrongJudgement,
              None
            )
          )
      }
    }

    // val label: ConstantTermLabel[N]
    val statement: F.Sequent =
      () |- F.Forall(
        out,
        Iff(
          equality(label.applySeq(vars), out),
          f
        )
      )

    library.last = Some(this)

  }

  /**
   * Allows to make definitions "by equality" of a function symbol
   */
  class SimpleFunctionDefinition[N <: F.Arity](using om: OutputManager)(fullName: String, line: Int, file: String)(
      val lambda: LambdaExpression[Term, Term, N],
      out: F.Variable,
      j: JUSTIFICATION
  ) extends FunctionDefinition[N](fullName, line, file)(lambda.bounds.asInstanceOf, out, out === lambda.body, j) {}

  object SimpleFunctionDefinition {
    def apply[N <: F.Arity](using om: OutputManager)(fullName: String, line: Int, file: String)(lambda: LambdaExpression[Term, Term, N]): SimpleFunctionDefinition[N] = {
      val intName = "definition_" + fullName
      val out = Variable(freshId(lambda.allSchematicLabels.map(_.id), "y"))
      val defThm = THM(F.ExistsOne(out, out === lambda.body), intName, line, file, InternalStatement)({
        have(SimpleDeducedSteps.simpleFunctionDefinition(lambda, out))
      })
      new SimpleFunctionDefinition[N](fullName, line, file)(lambda, out, defThm)
    }
  }

  class PredicateDefinition[N <: F.Arity](using om: OutputManager)(val fullName: String, line: Int, file: String)(val lambda: LambdaExpression[Term, Formula, N]) extends DEFINITION(line, file) {

    lazy val vars: Seq[F.Variable] = lambda.bounds.asInstanceOf
    val arity = lambda.arity

    lazy val label: ConstantAtomicLabelOfArity[N] = {
      (
        if (vars.length == 0) F.ConstantFormula(name)
        else F.ConstantPredicateLabel[N](name, vars.length.asInstanceOf[N])
      ).asInstanceOf[ConstantAtomicLabelOfArity[N]]
    }

    val innerJustification: theory.PredicateDefinition = {
      import lisa.utils.K.{predicateDefinition, findUndefinedSymbols}
      val underlambda = lambda.underlyingLTF
      val ulabel = K.ConstantFunctionLabel(name, vars.size)
      val undervars = vars.map(_.asInstanceOf[F.Variable].underlyingLabel)
      val judgement = theory.predicateDefinition(name, lambda.underlyingLTF)
      judgement match {
        case K.ValidJustification(just) =>
          just
        case wrongJudgement: K.InvalidJustification[?] =>
          if (!theory.belongsToTheory(underlambda.body)) {
            om.lisaThrow(
              UserInvalidDefinitionException(
                name,
                s"All symbols in the definition must belong to the theory. The symbols ${theory.findUndefinedSymbols(underlambda.body)} are unknown and you need to define them first."
              )
            )
          }
          if (!theory.isAvailable(ulabel)) {
            om.lisaThrow(UserInvalidDefinitionException(name, s"The symbol ${name} has already been defined and can't be redefined."))
          }
          if (!(underlambda.body.freeSchematicTermLabels.subsetOf(undervars.toSet) && underlambda.body.schematicFormulaLabels.isEmpty)) {
            om.lisaThrow(
              UserInvalidDefinitionException(
                name,
                s"The definition is not allowed to contain schematic symbols or free variables." +
                  s"Kernel returned error: The symbols {${(underlambda.body.freeSchematicTermLabels -- undervars.toSet ++ underlambda.body.freeSchematicTermLabels)
                      .mkString(", ")}} are free in the expression ${underlambda.body.toString}."
              )
            )
          }
          om.lisaThrow(
            LisaException.InvalidKernelJustificationComputation(
              "The final proof was rejected by LISA's logical kernel. This may be due to a faulty proof computation or an error in LISA.",
              wrongJudgement,
              None
            )
          )
      }
    }

    val statement: F.Sequent = () |- Iff(label.applySeq(vars), lambda.body)
    library.last = Some(this)
  }

  /////////////////////////
  //  Local Definitions  //
  /////////////////////////

  import lisa.utils.parsing.FOLPrinter.prettySCProof
  import lisa.utils.KernelHelpers.apply

  /**
   * A term with a definition, local to a proof.
   *
   * @param proof
   * @param id
   */
  abstract class LocalyDefinedVariable(val proof: library.Proof, id: Identifier) extends Variable(id) {

    val definition: proof.Fact
    lazy val definingFormula = proof.sequentOfFact(definition).right.head

    // proof.addDefinition(this, defin(this), fact)
    // val definition: proof.Fact = proof.getDefinition(this)
  }

  /**
   * A witness for a statement of the form ∃(x, P(x)) is a (fresh) variable y such that P(y) holds. This is a local definition, typically used with `val y = witness(fact)`
   * where `fact` is a fact of the form `∃(x, P(x))`. The property P(y) can then be used with y.elim
   */
  def witness(using _proof: library.Proof, line: sourcecode.Line, file: sourcecode.File, name: sourcecode.Name)(fact: _proof.Fact): LocalyDefinedVariable { val proof: _proof.type } = {

    val (els, eli) = _proof.sequentAndIntOfFact(fact)
    els.right.head match
      case Exists(x, inner) =>
        val id = freshId((els.allSchematicLabels ++ _proof.lockedSymbols ++ _proof.possibleGoal.toSet.flatMap(_.allSchematicLabels)).map(_.id), name.value)
        val c: LocalyDefinedVariable = new LocalyDefinedVariable(_proof, id) { val definition = assume(using proof)(inner.substitute(x := this)) }
        val defin = c.definingFormula
        val definU = defin.underlying
        val exDefinU = K.Exists(c.underlyingLabel, definU)

        if els.right.size != 1 || !K.isSame(els.right.head.underlying, exDefinU) then
          throw new UserInvalidDefinitionException(c.id, "Eliminator fact for " + c + " in a definition should have a single formula, equivalent to " + exDefinU + ", on the right side.")

        _proof.addElimination(
          defin,
          (i, sequent) =>
            val resSequent = (sequent.underlying -<< definU)
            List(
              SC.LeftExists(resSequent +<< exDefinU, i, definU, c.underlyingLabel),
              SC.Cut(resSequent ++<< els.underlying, eli, i + 1, exDefinU)
            )
        )

        c.asInstanceOf[LocalyDefinedVariable { val proof: _proof.type }]

      case _ => throw new Exception("Pick is used to obtain a witness of an existential statement.")

  }

}
