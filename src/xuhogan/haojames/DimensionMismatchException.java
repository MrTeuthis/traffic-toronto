package xuhogan.haojames;

/**
 * Thrown when a matrix operation is impossible due to a dimension mismatch. 
 * @author haosy
 *
 */
public class DimensionMismatchException extends Exception{
	private static final long serialVersionUID = -7531937569150773540L;
	
	/**
	 * Creates an empty DImensionMismatchException. 
	 */
	public DimensionMismatchException()
	{
	}
	
	/**
	 * Creates a DImensionMismatchException with a message. 
	 * @param message the message
	 */	
	public DimensionMismatchException(String message) {
		super(message);
	}
	
	/**
	 * Creates a DImensionMismatchException with a cause. 
	 * @param cause the cause
	 */
	public DimensionMismatchException(Throwable cause) {
		super(cause);
	}
	
	/**
	 * Creates a DImensionMismatchException with a message and a cause.
	 * @param message the message
	 * @param cause the cause
	 */
	public DimensionMismatchException(String message, Throwable cause) {
		super(message, cause);
	}
	
	/**
	 * Creates a DImensionMismatchException with a message, cause, enablesuppression, and writablestacktrace. 
	 * @param message the message
	 * @param cause the cause
	 * @param enableSuppression whether to enable suppression
	 * @param writableStackTrace whether the stack trace should be writable
	 */
	public DimensionMismatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
