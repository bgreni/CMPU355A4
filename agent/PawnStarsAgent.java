package agent;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.Throwable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import expert_iteration.ExItExperience;
import expert_iteration.ExItExperience.ExItExperienceState;
import expert_iteration.ExpertPolicy;
import game.Game;
import main.FileHandling;
import main.collections.FVector;
import main.collections.FastArrayList;
import metadata.ai.heuristics.Heuristics;
import metadata.ai.heuristics.terms.HeuristicTerm;
import metadata.ai.heuristics.terms.Material;
import metadata.ai.heuristics.terms.MobilitySimple;
import util.Context;
import util.Move;
import util.Trial;
import util.state.State;
import utils.AIUtils;

/**
 * alphabeta search agent based on the example from the Ludii AI repo
 * https://github.com/Ludeme/LudiiAI/blob/master/AI/src/search/minimax/AlphaBetaSearch.java
 */
public class PawnStarsAgent extends ExpertPolicy
{
	// init alpha as negative infinity
	private static final float ALPHA_INIT = Float.NEGATIVE_INFINITY;

	// init beta as positive infinity
	private static final float BETA_INIT = -ALPHA_INIT;
	
	/** unused (I think) */
	private static final float PARANOID_OPP_WIN_SCORE = 10000.f;
	
	/** We skip computing heuristics with absolute weight value lower than this
	* TODO: maybe try tuning this value? */
	public static final float ABS_HEURISTIC_WEIGHT_THRESHOLD = 0.01f;
	
	/** Our heuristic value function estimator
	TODO: set to some default in initAI(), so maybe try something else? */
	private Heuristics heuristicValueFunction = null;
	
	/** If true, we read our heuristic function to use from game's metadata */
	private final boolean heuristicsFromMetadata = true;
	
	/** We'll automatically return our move after at most this number of seconds if we only have one move */
	protected double autoPlaySeconds = 0.0;
	
	/** Estimated score of the root node based on last-run search */
	protected float estimatedRootScore = 0.f;
	
	/** The maximum heuristic eval we have ever observed */
	protected float maxHeuristicEval = 0.f;
	
	/** The minimum heuristic eval we have ever observed */
	protected float minHeuristicEval = 0.f;
	
	/** String to print to Analysis tab of the Ludii app */
	protected String analysisReport = null;
	
	/** Current list of moves available in root */
	protected FastArrayList<Move> currentRootMoves = null;
	
	/** The last move we returned. Need to memorise this for Expert Iteration with AlphaBeta */
	protected Move lastReturnedMove = null;
	
	/** Root context for which we've last performed a search */
	protected Context lastSearchedRootContext = null;
	
	/** Value estimates of moves available in root */
	protected FVector rootValueEstimates = null;
	
	/** The number of players in the game we're currently playing */
	protected int numPlayersInGame = 2;
	
	/** Remember if we proved a win in one of our searches */
	protected boolean provedWin = false;
	
	/** Needed for visualisations */
	protected float rootAlphaInit = ALPHA_INIT;
	
	/** Needed for visualisations */
	protected float rootBetaInit = BETA_INIT;
	
	/** Sorted (hopefully cleverly) list of moves available in root node */
	protected FastArrayList<Move> sortedRootMoves = null;
	
	/** If true at end of a search, it means we searched full tree (probably proved a draw) */
	protected boolean searchedFullTree = false;
//	private final int numPlayers = 2;
	
	public PawnStarsAgent()
	{
		friendlyName = "PawnStars";
	}
	
	@Override
	public Move selectAction
	(
		final Game game, 
		final Context context, 
		final double maxSeconds,
		final int maxIterations,
		final int maxDepth
	)
	{

		if (maxSeconds <= 0) {
			// complain if max seconds isn't greater than zero
			throw new MaxSecondsNotSetException("maxSeconds not greater than zero");
		}

		provedWin = false;
		final int depthLimit = maxDepth > 0 ? maxDepth : Integer.MAX_VALUE;
		lastSearchedRootContext = context;
		
		final long startTime = System.currentTimeMillis();
		final long stopTime = startTime + (long) (maxSeconds * 1000);
		
		lastReturnedMove = iterativeDeepening(game, context, maxSeconds, depthLimit, 1);
		return lastReturnedMove;
	}

	private void shuffleMoves(Game game, Context context) {
		currentRootMoves = new FastArrayList<Move>(game.moves(context).moves());

		// Create a shuffled version of list of moves (random tie-breaking)
		final FastArrayList<Move> tempMovesList = new FastArrayList<Move>(currentRootMoves);
		sortedRootMoves = new FastArrayList<Move>(currentRootMoves.size());
		while (!tempMovesList.isEmpty())
		{
			sortedRootMoves.add(tempMovesList.removeSwap(ThreadLocalRandom.current().nextInt(tempMovesList.size())));
		}
	}

	private boolean shouldInterrupt(long stopTime) {
		return System.currentTimeMillis() >= stopTime || wantsInterrupt;
	}

	/**
	 * Runs iterative deepening alpha-beta
	 * @param game
	 * @param context
	 * @param maxSeconds
	 * @param maxDepth
	 * @param startDepth
,	 */
	public Move iterativeDeepening
	(
		final Game game, 
		final Context context, 
		final double maxSeconds, 
		final int maxDepth,
		final int startDepth
	)
	{
		final long startTime = System.currentTimeMillis();
		long stopTime = (maxSeconds > 0.0) ? startTime + (long) (maxSeconds * 1000) : Long.MAX_VALUE;

		shuffleMoves(game, context);
		
		final int numRootMoves = sortedRootMoves.size();
		final List<ScoredMove> scoredMoves = new ArrayList<ScoredMove>(sortedRootMoves.size());
		
		// Vector for visualisation purposes
		rootValueEstimates = new FVector(currentRootMoves.size());
		
		// storing scores found for purpose of move ordering
		final FVector moveScores = new FVector(numRootMoves);
		int searchDepth = startDepth - 1;
		final int maximisingPlayer = context.state().playerToAgent(context.state().mover());
		
		// best move found so far during a fully-completed search 
		// (ignoring incomplete early-terminated search)
		Move bestMoveCompleteSearch = sortedRootMoves.get(0);

		rootAlphaInit = ALPHA_INIT;
		rootBetaInit = BETA_INIT;
		
		while (searchDepth < maxDepth)
		{
			++searchDepth;
			searchedFullTree = true;

			float score = rootAlphaInit;
			float alpha = rootAlphaInit;
			final float beta = rootBetaInit;

			// best move during this particular search
			Move bestMove = sortedRootMoves.get(0);


//			final Context copyContext = new Context(context);
//			final Result best = BNS(copyContext, searchDepth, alpha, beta, maximisingPlayer, stopTime);
//			Move bestMove = null;
//			System.out.println("BEST: "  + best.score);
//			if (score < best.score) {
//				score = best.score;
//				bestMove = best.move;
//			}

//			if (score > alpha)
//				alpha = score;

			for (int i = 0; i < numRootMoves; ++i)
			{
				final Context copyContext = new Context(context);
				final Move m = sortedRootMoves.get(i);
				game.apply(copyContext, m);
				final float value = MTDF(copyContext, searchDepth - 1,  0,maximisingPlayer, stopTime);
//				final float value = alphaBeta(copyContext, searchDepth - 1, alpha, beta, maximisingPlayer, stopTime);
				if (shouldInterrupt(stopTime))	// time to abort search
				{
					bestMove = null;
					break;
				}

				final int origMoveIdx = currentRootMoves.indexOf(m);
				if (origMoveIdx >= 0)
				{
					rootValueEstimates.set(origMoveIdx, (float) scoreToValueEst(value, rootAlphaInit, rootBetaInit));
				}

				moveScores.set(i, value);

				if (value > score)		// new best move found
				{
					score = value;
					bestMove = m;
				}

				if (score > alpha)		// new lower bound
					alpha = score;

				if (alpha >= beta)		// beta cut-off
					break;
			}

			// alpha-beta is over, this is iterative deepening stuff again

			if (bestMove != null)		// search was not interrupted
			{
				estimatedRootScore = score;

				if (score == rootBetaInit)
				{
					// we've just proven a win, so we can return best move
					// found during this search
					analysisReport = friendlyName + " found a proven win at depth " + searchDepth + ".";
					provedWin = true;
					return bestMove;
				}
				else if (score == rootAlphaInit)
				{
					// we've just proven a loss, so we return the best move
					// of the PREVIOUS search (delays loss for the longest
					// amount of time)
					analysisReport = friendlyName + " found a proven loss at depth " + searchDepth + ".";
					return bestMoveCompleteSearch;
				}
				else if (searchedFullTree)
				{
					// We've searched full tree but did not prove a win or loss
					// probably means a draw, play best line we have
					analysisReport = friendlyName + " completed search of depth " + searchDepth + " (no proven win or loss).";
					return bestMove;
				}

				bestMoveCompleteSearch = bestMove;
			}
			else
			{
				// decrement because we didn't manage to complete this search
				--searchDepth;
			}

			if (shouldInterrupt(stopTime))
			{
				// we need to return
				analysisReport = friendlyName + " completed search of depth " + searchDepth + ".";
				return bestMoveCompleteSearch;
			}

			sortMoves(scoredMoves, moveScores, numRootMoves);

			// clear the vector of scores
			moveScores.fill(0, numRootMoves, 0.f);
		}

		analysisReport = friendlyName + " completed search of depth " + searchDepth + ".";
		return bestMoveCompleteSearch;
	}

	private void sortMoves(List<ScoredMove> scoredMoves, FVector moveScores, int numRootMoves) {
		// order moves based on scores found, for next search
		scoredMoves.clear();
		for (int i = 0; i < numRootMoves; ++i)
		{
			scoredMoves.add(new ScoredMove(sortedRootMoves.get(i), moveScores.get(i)));
		}
		Collections.sort(scoredMoves);

		sortedRootMoves.clear();
		for (int i = 0; i < numRootMoves; ++i)
		{
			sortedRootMoves.add(scoredMoves.get(i).move);
		}
	}

	public float MTDF(
			final Context context,
			final int depth,
			final float f,
			final int maximisingPlayer,
			final long stopTime
	) {
		float g = f;
		float upperBound = 100000000.f;
		float lowerBound = -upperBound;

		while (lowerBound < upperBound) {

			float beta = Math.max(g, lowerBound + 1);

			g = alphaBeta(context, depth, beta-1, beta, maximisingPlayer, stopTime);

			if (g < beta) {
				upperBound = g;
			} else {
				lowerBound = g;
			}

			if (shouldInterrupt(stopTime))
				return g;
		}
		return g;
	}

	public Result BNS(
			final Context context,
			final int depth,
			float alpha,
			float beta,
			final int maximisingPlayer,
			final long stopTime
	) {
		Game game = context.game();
		final FastArrayList<Move> moves = game.moves(context).moves();
		int subtreeCount = moves.size();
		int betterCount = 0;
		float bestVal = 0;
		Move bestMove = moves.get(0);

		do {
			betterCount = 0;
			float test = nextGuess(alpha, beta, (float)subtreeCount);
			for (int i = 0; i < subtreeCount; i++) {
				final Context copyContext = new Context(context);
				final Move m = moves.get(i);
				game.apply(copyContext, m);
				final float currVal = -alphaBeta(copyContext, depth - 1, -(test), -(test-1), maximisingPlayer, stopTime);
//				System.out.println("VAL: " + currVal + " TEST: " + test);
				if (currVal >= test) {
					System.out.println("GOOD MOVE FOUND");
					betterCount += 1;
					bestVal = currVal;
					bestMove = m;
				}

				if (shouldInterrupt(stopTime)) {
					System.out.println("Interrupted");
					return new Result(bestMove, bestVal);
				}
			}
			alpha = test;
			subtreeCount -= betterCount;

		} while (!(beta - alpha < 2 || betterCount == 1));
//		System.out.println("BrokeLoop");
		return new Result(bestMove, bestVal);
	}

	private float nextGuess(final float alpha, final float beta, final float subtreeCount) {
		return alpha + (beta - alpha) * (subtreeCount - 1) / subtreeCount;
	}

	
	/**
	 * Recursive alpha-beta search function.
	 * 
	 * @param context
	 * @param depth
	 * @param inAlpha
	 * @param inBeta
	 * @param maximisingPlayer Who is the maximising player?
	 * @param stopTime
	 * @return (heuristic) evaluation of the reached state, from perspective of maximising player.
	 */
	public float alphaBeta
	(
		final Context context,
		final int depth,
		final float inAlpha,
		final float inBeta,
		final int maximisingPlayer,
		final long stopTime
	)
	{
		final Trial trial = context.trial();
		final State state = context.state();
		
		if (trial.over() || !context.state().active(maximisingPlayer))
		{
			// terminal node (at least for maximising player)
			return (float) AIUtils.agentUtilities(context.state())[maximisingPlayer] * BETA_INIT;
		}
		else if (depth == 0)
		{
			return evalHeuristic(context, maximisingPlayer, state);
		}
		
		final Game game = context.game();
		final int mover = state.playerToAgent(state.mover());
		
		final FastArrayList<Move> legalMoves = game.moves(context).moves();
		final int numLegalMoves = legalMoves.size();
		float alpha = inAlpha;
		float beta = inBeta;
		
		if (mover == maximisingPlayer)
		{
			float score = ALPHA_INIT;
			
			for (int i = 0; i < numLegalMoves; ++i)
			{
				final Context copyContext = new Context(context);
				final Move m = legalMoves.get(i);
				game.apply(copyContext, m);
				final float value = alphaBeta(copyContext, depth - 1, alpha, beta, maximisingPlayer, stopTime);
				
				if (shouldInterrupt(stopTime))	// time to abort search
				{
					return 0;
				}

				if (value > score)
					score = value;

				if (score > alpha)
					alpha = score;

				if (alpha >= beta)	// beta cut-off
					break;
			}
			
			return score;
		}
		else
		{
			float score = BETA_INIT;
			
			for (int i = 0; i < numLegalMoves; ++i)
			{
				final Context copyContext = new Context(context);
				final Move m = legalMoves.get(i);
				game.apply(copyContext, m);
				final float value = alphaBeta(copyContext, depth - 1, alpha, beta, maximisingPlayer, stopTime);
				
				if (shouldInterrupt(stopTime))	// time to abort search
				{
					return 0;
				}

				if (value < score)
					score = value;
				
				if (score < beta)
					beta = score;
				
				if (alpha >= beta)	// alpha cut-off
					break;
			}
			
			return score;
		}
	}

	private float evalHeuristic(Context context, final int maximisingPlayer, State state) {
		searchedFullTree = false;

		// heuristic evaluation
		float heuristicScore = heuristicValueFunction.computeValue(
				context, maximisingPlayer, ABS_HEURISTIC_WEIGHT_THRESHOLD);

		final int opp = opponents(maximisingPlayer)[0];
		if (context.state().active(opp))
			heuristicScore -= heuristicValueFunction.computeValue(context, opp, ABS_HEURISTIC_WEIGHT_THRESHOLD);
		else if (context.state().winners().contains(opp))
			heuristicScore -= PARANOID_OPP_WIN_SCORE;

		// Invert scores if players swapped
		if (state.playerToAgent(maximisingPlayer) != maximisingPlayer)
			heuristicScore = -heuristicScore;

		minHeuristicEval = Math.min(minHeuristicEval, heuristicScore);
		maxHeuristicEval = Math.max(maxHeuristicEval, heuristicScore);

		return heuristicScore;
	}
	
	/**
	 * @param player
	 * @return Opponents of given player
	 */
	public int[] opponents(final int player)
	{
		final int[] opponents = new int[numPlayersInGame - 1];
		int idx = 0;
		
		for (int p = 1; p <= numPlayersInGame; ++p)
		{
			if (p != player)
				opponents[idx++] = p;
		}
		
		return opponents;
	}
	
	/**
	 * Converts a score into a value estimate in [-1, 1]. Useful for visualisations.
	 * 
	 * @param score
	 * @param alpha 
	 * @param beta 
	 * @return Value estimate in [-1, 1] from unbounded (heuristic) score.
	 */
	public double scoreToValueEst(final float score, final float alpha, final float beta)
	{
		if (score == alpha)
			return -1.0;
		
		if (score == beta)
			return 1.0;
		
		// Map to range [-0.8, 0.8] based on most extreme heuristic evaluations
		// observed so far.
		return -0.8 + (0.8 - -0.8) * ((score - minHeuristicEval) / (maxHeuristicEval - minHeuristicEval));
	}
	
	//-------------------------------------------------------------------------
	
	@Override
	public void initAI(final Game game, final int playerID)
	{
		if (heuristicsFromMetadata)
		{
			// Read heuristics from game metadata
			final metadata.ai.Ai aiMetadata = game.metadata().ai();
			if (aiMetadata != null && aiMetadata.heuristics() != null)
			{
				heuristicValueFunction = aiMetadata.heuristics();
			}
			else
			{
				// construct default heuristic
				heuristicValueFunction = new Heuristics(new HeuristicTerm[]{
						new Material(null, Float.valueOf(1.f), null),
						new MobilitySimple(null, Float.valueOf(0.001f))
				});
			}
		}
		
		if (heuristicValueFunction != null)
			heuristicValueFunction.init(game);
		
		// reset these things used for visualisation purposes
		estimatedRootScore = 0.f;
		maxHeuristicEval = 0.f;
		minHeuristicEval = 0.f;
		analysisReport = null;
		
		currentRootMoves = null;
		rootValueEstimates = null;
		
		// and these things for ExIt
		lastSearchedRootContext = null;
		lastReturnedMove = null;
		
		// This will only be used for breakthrough so it will only work for two player games
		numPlayersInGame = 2;
	}
	
	@Override
	public boolean supportsGame(final Game game)
	{
		if (game.players().count() != 2)
			return false;
		
		if (game.isStochasticGame())
			return false;
		
		if (game.hiddenInformation())
			return false;
		
		return game.isAlternatingMoveGame();
	}
	
	@Override
	public double estimateValue()
	{
		return scoreToValueEst(estimatedRootScore, rootAlphaInit, rootBetaInit);
	}
	
	@Override
	public String generateAnalysisReport()
	{
		return analysisReport;
	}
	
	@Override
	public AIVisualisationData aiVisualisationData() {
		if (currentRootMoves == null || rootValueEstimates == null)
			return null;

		final FVector aiDistribution = rootValueEstimates.copy();
		aiDistribution.subtract(aiDistribution.min());

		return new AIVisualisationData(aiDistribution, rootValueEstimates, currentRootMoves);
	}
	
	@Override
	public FastArrayList<Move> lastSearchRootMoves()
	{
		final FastArrayList<Move> moves = new FastArrayList<Move>(currentRootMoves.size());
		for (final Move move : currentRootMoves)
		{
			moves.add(move);
		}
		return moves;
	}
	
	@Override
	public FVector computeExpertPolicy(final double tau)
	{
		final FVector distribution = FVector.zeros(currentRootMoves.size());
		distribution.set(currentRootMoves.indexOf(lastReturnedMove), 1.f);
		distribution.softmax();
		return distribution;
	}
	
	@Override
	public ExItExperience generateExItExperience()
	{
    	return new ExItExperience
    			(
    				new ExItExperienceState(lastSearchedRootContext.trial()),
    				currentRootMoves,
    				computeExpertPolicy(1.0),
    				FVector.zeros(currentRootMoves.size())
    			);
	}
	
	//-------------------------------------------------------------------------
	
	/**
	 * Wrapper for score + move, used for sorting moves based on scores.
	 * 
	 * @author Dennis Soemers
	 */
	private class ScoredMove implements Comparable<ScoredMove>
	{
		/** The move */
		public final Move move;
		/** The move's score */
		public final float score;
		
		/**
		 * Constructor
		 * @param move
		 * @param score
		 */
		public ScoredMove(final Move move, final float score)
		{
			this.move = move;
			this.score = score;
		}

		@Override
		public int compareTo(final ScoredMove other)
		{
			final float delta = other.score - this.score;
			if (delta < 0.f)
				return -1;
			else if (delta > 0.f)
				return 1;
			else
				return 0;
		}
	}

	private class MaxSecondsNotSetException extends RuntimeException {
		public MaxSecondsNotSetException(String message) {
			super(message);
		}
	}

	private class Result {
		public Move move;
		public float score;

		public Result(Move move, float score) {
			this.move = move;
			this.score = score;
		}
	}
}