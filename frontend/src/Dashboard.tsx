import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Grid,
  Paper,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import axios from 'axios';
import { API_BASE_URL } from './config/api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

// Types
interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
}

interface PnL {
  total_pnl: number;
  timestamp: string;
  positions_count: number;
  error?: string;
}

const Dashboard: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [pnl, setPnL] = useState<PnL | null>(null);
  const [isTrading, setIsTrading] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('');

  // Fetch positions and PnL periodically
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [positionsRes, pnlRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/positions`),
          axios.get(`${API_BASE_URL}/api/pnl`),
        ]);
        setPositions(positionsRes.data);
        setPnL(pnlRes.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Start/Stop trading
  const handleTrading = async () => {
    try {
      const endpoint = isTrading ? '/api/stop' : '/api/start';
      await axios.post(`${API_BASE_URL}${endpoint}`);
      setIsTrading(!isTrading);
    } catch (error) {
      console.error('Error toggling trading:', error);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          AI Trading Dashboard
        </Typography>

        {/* Controls */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Button
                variant="contained"
                color={isTrading ? 'error' : 'success'}
                onClick={handleTrading}
                fullWidth
              >
                {isTrading ? 'Stop Trading' : 'Start Trading'}
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Symbol</InputLabel>
                <Select
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                >
                  <MenuItem value="NIFTY">NIFTY</MenuItem>
                  <MenuItem value="BANKNIFTY">BANKNIFTY</MenuItem>
                  <MenuItem value="RELIANCE">RELIANCE</MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Grid>
        </Grid>

        {/* P&L Display */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Total P&L
              </Typography>
              <Typography
                variant="h4"
                color={(pnl?.total_pnl ?? 0) >= 0 ? 'success.main' : 'error.main'}
              >
                ₹{pnl?.total_pnl?.toFixed(2) ?? '0.00'}
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {/* Positions Table */}
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Open Positions
              </Typography>
              {positions.length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Quantity</th>
                      <th>Avg Price</th>
                      <th>Current Price</th>
                      <th>P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position) => (
                      <tr key={position.symbol}>
                        <td>{position.symbol}</td>
                        <td>{position.quantity}</td>
                        <td>₹{position.avgPrice.toFixed(2)}</td>
                        <td>₹{position.currentPrice.toFixed(2)}</td>
                        <td
                          style={{
                            color: position.pnl >= 0 ? 'green' : 'red',
                          }}
                        >
                          ₹{position.pnl.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Typography color="textSecondary">No open positions</Typography>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default Dashboard;